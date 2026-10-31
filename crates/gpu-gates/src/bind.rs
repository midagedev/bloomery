//! The server's engine over a [`Seat`]: [`SeatEngine`] implements
//! `serve::Engine`. The file's vocabulary ([`Vocab`]), its chat surface
//! ([`ChatSurface`]) and the sampler factory ([`sampler_factory`]) are the
//! serve crate's, every backend's, and re-exported here for the seats.
//!
//! The seat — the model, its draft if one runs, and the body's rules — lives
//! on one thread of its own, for the whole run. The server calls its engine
//! from whichever connection thread holds the slot; the model's device state
//! and the pinned dispatcher slot belong to the thread that opened them, so
//! every step runs on that thread and the engine is a handle that sends it
//! commands. The opener comes in as a closure, and the seat as a trait this
//! library leaves open (the serve binaries' seat tree, `shared/serve_seats/`,
//! holds its implementers, the device crates' seats), because this module
//! names no device crate (see [`crate::generate`]) or the session crate.
//!
//! How long a prefix of the cache can be kept, and why no longer, is the
//! seat's rule, which the engine thread answers ([`SeatEngine`]'s `keepable`
//! and `keep_limit`); so are the prompt feed, a pass of the draft
//! ([`Seat::pass`]), where a prompt call is cut, and the state the server's
//! prompt cache saves and puts back ([`Seat::snapshot`], [`Seat::resume`]; a
//! seat that copies on a stream of its own begins a save that does not stop the
//! slots, [`Seat::snapshot_begin`], and puts back a prefix,
//! [`Seat::resume_prefix`]). How
//! far the positions go is the lesser of the cache and the positions the body
//! computes the model at (`spawn`'s `defined`): the server refuses a request
//! past it, so a client's long prompt or `max_tokens` never reaches the body's
//! refusal, which is fatal.
//!
//! The seats whose state is a `bloomery_gpu` `SeqView` share one owner of what
//! the prompt cache holds and waits for: [`Cached`] is their [`Saved`] (the
//! body's state with the seat's own side), [`begin`] their
//! [`Seat::snapshot_begin`] over the body's save on its lane (a pool with no
//! room is the server's `NoRoom`, [`seat_failure`]), and [`Cached::body`] the
//! one way to read a state, which waits for its copy to land first.
//!
//! A sampled token's logits cross to the calling thread in one buffer the
//! engine owns: lent with each `Next` that wants them, filled on the engine
//! thread, and handed back with the reply, so no token allocates a row.
//!
//! A slot file's state ([`Seat::save_state`], [`Seat::restore_state`])
//! crosses between the server's thread and the engine thread through a pipe
//! of chunks of `PIPE_CHUNK` bytes, at most `PIPE_DEPTH` of them waiting,
//! never staged whole: past the seat's own copy the host holds a few chunks.
//! The seat's error kind crosses with the answer, so only an engine failure
//! ends the server.
//!
//! What `/props` says about the engine ([`model_props`], [`placement_props`])
//! is read from the file's header and the placement plan the engine loads by,
//! once, before the load; the engine hands it back unchanged.

use std::any::Any;
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bloomery_gpu::GpuError;
use bloomery_gpu::checkpoint::saved::{Pending as Saving, SeqView};
use bloomery_gpu::checkpoint::staging::NoRoom;
use bloomery_levers::Levers;
use gguf::Split;
use model::placement::workstation::{self, DeviceId, HostRead};
use model::placement::{Device, Machine, ModelTensors, Plan, Role};
use serve::media::{MediaFeed, SharedMediaModel};
use serve::{
    CacheNote, DeviceProps, Drafted, Engine, EngineError, EngineProps, FATAL_LINGER, Flight,
    Landed, ModelProps, Pending, PlacementProps, ResidencyReset, Saved, SavedState, ServeError,
    Server, ServerConfig, SharedPrefix, SlotConfig, StateError, Tokenizer, Waits,
};

use crate::GateError;
use crate::record::Record;
use crate::side_seat::SideEngine;

pub use serve::sampling::{sampler_factory, sampler_params};
pub use serve::vocab::{ChatSurface, TemplateFile, Vocab};

/// What a seat's server binds with, past its engine.
pub struct Listen<'a> {
    pub host: &'a str,
    pub port: u16,
    /// `model` in responses and the `/v1/models` id.
    pub alias: String,
    /// The model file, `/props`' `model_path`.
    pub path: &'a Path,
    pub template: String,
    /// The family's request preparation; `None` renders the request as the
    /// template reads it.
    pub preparation: Option<serve::ChatPreparation>,
    /// `--slot-save-path`; `None` refuses every slot action.
    pub slot_save_path: Option<PathBuf>,
    pub api_keys: serve::flag::ApiKeys,
    /// The slots the engine made at its open, which the server steps together.
    pub parallel: usize,
    pub queue_depth: Option<usize>,
    /// Whether a request that follows a program feeds each run of ids the program fixes as one
    /// prompt call (`Levers::schema_jump`).
    pub schema_jump: bool,
    /// The embedding model the server holds beside the engine (`--embd-model`), which its
    /// embeddings routes answer from; `None` answers them with a 501.
    pub side: Option<SideEngine>,
}

/// The server over a seat's `engine`: the sampler crate's chain, the fatal
/// linger every seat keeps, the seat's own slots as the server's, and the side
/// model beside it. The seat prints its own `listening` record from the bound
/// server and runs it.
pub fn bind_server(engine: SeatEngine, l: Listen<'_>) -> Result<Server, ServeError> {
    let slots = slot_config(&l);
    let config = ServerConfig {
        model_alias: l.alias,
        model_path: l.path.display().to_string(),
        chat_template: l.template,
        chat_preparation: l.preparation,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: l.slot_save_path,
        api_keys: l.api_keys,
    };
    // A start by an agent bound its listener before the fetch ([`crate::start::bind_early`]);
    // the server takes that socket over, so the port never unbinds.
    let server = match crate::start::take_early(l.host, l.port).map_err(ServeError::Io)? {
        Some(listener) => Server::bind_listener(listener, Box::new(engine), config, slots)?,
        None => Server::bind_with((l.host, l.port), Box::new(engine), config, slots)?,
    };
    let server = match crate::start::noted_place() {
        Some(place) => server.with_place(place),
        None => server,
    };
    Ok(match l.side {
        Some(side) => server.with_side(Box::new(side)),
        None => server,
    })
}

/// The server's slots over a seat's [`Listen`]: its count and queue depth, and
/// whether a request that follows a program feeds each run of ids the program
/// fixes as one prompt call (`BLOOMERY_SCHEMAJUMP`).
fn slot_config(l: &Listen<'_>) -> SlotConfig {
    SlotConfig {
        parallel: l.parallel,
        queue_depth: l.queue_depth,
        schema_jump: l.schema_jump,
        ..SlotConfig::default()
    }
}

/// What set a seat's slot count, the word its `parallel` line's `from=` names:
/// the one owner of those words, which the gates read as text. A seat holds the
/// type, never the word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotsFrom {
    /// `--parallel` as given.
    Flag,
    /// No `--parallel` beside a set `--ctx`: one slot at the whole context.
    Ctx,
    /// Neither flag: the seat's own default count over its automatic context.
    Default,
    /// A route trace records one sequence's steps, so one slot is served.
    RouteTrace,
    /// The DSpark draft's window is one sequence's, so one slot is served.
    Dspark,
    /// `--parallel auto`: the most slots the plan holds at one request's context.
    Auto,
}

impl SlotsFrom {
    /// The `from=` word.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            SlotsFrom::Flag => "flag",
            SlotsFrom::Ctx => "ctx",
            SlotsFrom::Default => "default",
            SlotsFrom::RouteTrace => "route-trace",
            SlotsFrom::Dspark => "dspark",
            SlotsFrom::Auto => "auto",
        }
    }
}

impl std::fmt::Display for SlotsFrom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.word())
    }
}

/// `--parallel` (`-np`: a count, or `auto`) and `--slot-ctx` (one request's context) as a
/// seat's parse takes them ([`SlotFlags::take`]), read beside `--ctx` once the parse ends
/// ([`SlotFlags::ask`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotFlags {
    parallel: Option<Parallel>,
    slot_ctx: Option<usize>,
}

/// `--parallel`'s value.
#[derive(Clone, Copy, Debug)]
enum Parallel {
    Count(usize),
    Auto,
}

/// What the slot flags ask beside `--ctx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotAsk {
    /// The `--ctx` total the slots split and the `--parallel` count, each `None` where unset:
    /// `--slot-ctx S` asks a total of `S × N` over `--parallel N`, one slot when it names none.
    Given {
        ctx: Option<usize>,
        parallel: Option<usize>,
    },
    /// `--parallel auto`.
    Auto(AutoSlots),
}

/// `--parallel auto`: the most slots, up to `ctx_ask::AUTO_MOST`, whose plan holds one
/// request's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoSlots {
    /// `--slot-ctx`; unset, the context one slot's default takes.
    pub slot_ctx: Option<usize>,
}

impl SlotFlags {
    /// Takes `flag` with its value `v` when it is a slot flag (`Ok(true)`); `Ok(false)` leaves
    /// it to the caller.
    pub fn take(&mut self, flag: &str, v: &str) -> Result<bool, GateError> {
        match flag {
            "--parallel" | "-np" if v == "auto" => self.parallel = Some(Parallel::Auto),
            "--parallel" | "-np" => {
                self.parallel = Some(Parallel::Count(serve::flag::number(flag, v)?));
            }
            serve::flag::SLOT_CTX => self.slot_ctx = Some(serve::flag::number(flag, v)?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The ask beside `ctx` (`--ctx`, the slots' total). Refused by name: `--parallel 0`,
    /// `--slot-ctx 0`, `--slot-ctx` beside `--ctx`, `--parallel auto` beside `--ctx`, and a
    /// total past `usize`. Notes whether `--slot-ctx` was given for the start's refusals
    /// ([`crate::start::note_slot_ctx`]); each call overwrites the note.
    pub fn ask(self, ctx: Option<usize>) -> Result<SlotAsk, GateError> {
        crate::start::note_slot_ctx(self.slot_ctx.is_some());
        let parallel = match self.parallel {
            Some(Parallel::Count(0)) => return Err(NoSlot.into()),
            Some(Parallel::Count(n)) => Some(n),
            Some(Parallel::Auto) | None => None,
        };
        let auto = matches!(self.parallel, Some(Parallel::Auto));
        match (self.slot_ctx, ctx) {
            (Some(0), _) => Err("--slot-ctx 0: a slot holds no position".into()),
            (Some(s), Some(c)) => Err(format!(
                "--slot-ctx {s} beside --ctx-size {c}: --slot-ctx is one request's context and \
                 --ctx-size the total the slots split; give one of them"
            )
            .into()),
            (None, Some(c)) if auto => Err(format!(
                "--parallel auto beside --ctx-size {c}: auto counts the slots one request's \
                 context leaves room for; give that context as --slot-ctx, not the slots' total"
            )
            .into()),
            (slot_ctx, _) if auto => Ok(SlotAsk::Auto(AutoSlots { slot_ctx })),
            (Some(s), None) => {
                let n = parallel.unwrap_or(1);
                let total = s.checked_mul(n).ok_or_else(|| {
                    format!("--slot-ctx {s} over --parallel {n}: the total passes usize")
                })?;
                Ok(SlotAsk::Given {
                    ctx: Some(total),
                    parallel,
                })
            }
            (None, ctx) => Ok(SlotAsk::Given { ctx, parallel }),
        }
    }

    /// [`SlotFlags::ask`] for a seat that serves the count `--parallel` names: the `--ctx`
    /// total and the count, `--parallel auto` refused by name.
    pub fn given(self, ctx: Option<usize>) -> Result<(Option<usize>, Option<usize>), GateError> {
        match self.ask(ctx)? {
            SlotAsk::Given { ctx, parallel } => Ok((ctx, parallel)),
            SlotAsk::Auto(_) => Err("--parallel auto: this seat serves the count --parallel \
                 names; the qwen38 seat counts its slots from memory"
                .into()),
        }
    }
}

/// A slot count of none: a seat's own parse refuses `--parallel 0` before
/// [`slots_of`], which refuses it too, so no call serves a count of no slot.
#[derive(Debug, PartialEq, Eq)]
pub struct NoSlot;

impl std::fmt::Display for NoSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("--parallel 0: the server serves no slot")
    }
}

impl std::error::Error for NoSlot {}

/// The resident sequences a seat serves and what set the count
/// ([`SlotsFrom`]): `parallel` (`--parallel`) as given
/// ([`SlotsFrom::Flag`]); unset beside a set `--ctx`, one slot at the whole
/// context — a `--ctx-size` is what one request gets, as llama-server reads it
/// ([`SlotsFrom::Ctx`]); unset with no context named, the seat's own default
/// (`seat_default`) over its automatic context ([`SlotsFrom::Default`]).
///
/// # Errors
/// [`NoSlot`] when the count comes out as zero.
pub fn slots_of(
    parallel: Option<usize>,
    ctx_set: bool,
    seat_default: usize,
) -> Result<(usize, SlotsFrom), NoSlot> {
    let (slots, from) = match parallel {
        Some(n) => (n, SlotsFrom::Flag),
        None if ctx_set => (1, SlotsFrom::Ctx),
        None => (seat_default, SlotsFrom::Default),
    };
    if slots == 0 {
        return Err(NoSlot);
    }
    Ok((slots, from))
}

/// The slots a seat serves and what set the count (`from`, as [`slots_of`]
/// returns it); a set `--ctx` with no
/// `--parallel` prints the one line that says the flag is one request's
/// context ([`one_request_line`]).
pub fn slots_given(
    parallel: Option<usize>,
    ctx: Option<usize>,
    default: usize,
) -> Result<(usize, SlotsFrom), GateError> {
    let (slots, from) = slots_of(parallel, ctx.is_some(), default)?;
    if from == SlotsFrom::Ctx {
        eprintln!(
            "{}",
            one_request_line(ctx.unwrap_or_default(), crate::start::slot_ctx_noted())
        );
    }
    Ok((slots, from))
}

/// The line a count of one from a set context ([`SlotsFrom::Ctx`]) prints, naming the context
/// flag the slot parser read (`per_slot`, as [`crate::start::slot_ctx_noted`] keeps it): N slots
/// split a `--ctx-size` total, and each takes a whole `--slot-ctx`.
#[must_use]
pub fn one_request_line(ctx: usize, per_slot: bool) -> String {
    if per_slot {
        format!(
            "--slot-ctx {ctx} is one request's context; add --parallel N to serve N requests at \
             once (each takes it)"
        )
    } else {
        format!(
            "--ctx-size {ctx} is one request's context; add --parallel N to serve N requests at \
             once (they split it)"
        )
    }
}

/// Checks that an explicit context leaves at least one position in each requested slot.
/// An unset slot count can serve one slot, and an unset context is sized by the plan.
pub fn check_slot_ctx(ctx: Option<usize>, parallel: Option<usize>) -> Result<(), GateError> {
    if let Some(total) = ctx {
        model::placement::slots::split_ctx(
            u64::try_from(total)?,
            u64::try_from(parallel.unwrap_or(1))?,
            std::num::NonZeroU64::MIN,
        )
        .map_err(|e| {
            format!(
                "--ctx-size {total} --parallel {}: {e}",
                parallel.unwrap_or(1)
            )
        })?;
    }
    Ok(())
}

/// A serving seat's `--place` word, the one reader of the placement grammar:
/// [`Place::parse`](crate::generate::Place::parse) (`a`, `gate`, `bp`, `tp`, or a card
/// list `<stage>[+<tier>…]` of card names and device ordinals). `serves` is the
/// `(body, tiers)` of a body that serves a fixed count of expert tier cards: a
/// list of more is refused by name here, before any plan. `None` leaves the
/// count to the seat's plan, which refuses a tier its body cannot serve.
pub fn place_word(
    v: &str,
    serves: Option<(&str, usize)>,
) -> Result<crate::generate::Place, GateError> {
    let place = crate::generate::Place::parse(v)?;
    if let Some((body, tiers)) = serves {
        place.serves(body, tiers)?;
    }
    Ok(place)
}

/// The address a serving seat listens on unless `--host` and `--port` say another: the one
/// owner of the default, which the start's early listener ([`crate::start::bind_early`]) shares.
pub const DEFAULT_HOST: &str = "127.0.0.1";
/// See [`DEFAULT_HOST`].
pub const DEFAULT_PORT: u16 = 8080;

/// The flags every serving seat takes beside its own: the listener (`--host`,
/// `--port`, `--queue-depth`), the keys (`--api-key`, `--api-key-file`), the
/// context (`--ctx`, under every spelling `serve::flag::CTX` lists), the
/// placement word (`--place`, read by [`place_word`]), the alias, the chat
/// template file, the slot actions' directory (`--slot-save-path`),
/// `--parallel` (and `-np`) with `--slot-ctx` ([`SlotFlags`]), and `--plan`.
pub struct SeatFlags {
    pub host: String,
    pub port: u16,
    pub ctx: Option<usize>,
    pub place: Option<crate::generate::Place>,
    pub alias: Option<String>,
    /// `--chat-template-file`, replacing the file's own template.
    pub template_file: TemplateFile,
    /// `--plan`: the records before the load, then exit.
    pub plan_only: bool,
    pub parallel: Option<usize>,
    pub queue_depth: Option<usize>,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    pub api_keys: serve::flag::ApiKeys,
    /// `--slot-save-path`: the directory the slot actions answer from; `None`
    /// refuses every one.
    pub slot_save_path: Option<PathBuf>,
}

impl SeatFlags {
    /// The flags of `args`. A flag that is not one of the common set goes to
    /// `own` with its value: `Ok(true)` took it, `Ok(false)` is an unknown
    /// argument, refused by name with `usage`. `--help` returns `usage` as
    /// the error; `--ctx 0`, `--parallel 0` and `--parallel auto` are refused by name
    /// ([`SlotFlags::given`]).
    pub fn parse(
        args: &[String],
        usage: &str,
        mut own: impl FnMut(&str, &str) -> Result<bool, GateError>,
    ) -> Result<SeatFlags, GateError> {
        let mut a = SeatFlags {
            host: DEFAULT_HOST.to_owned(),
            port: DEFAULT_PORT,
            ctx: None,
            place: None,
            alias: None,
            template_file: TemplateFile::default(),
            plan_only: false,
            parallel: None,
            queue_depth: None,
            api_keys: serve::flag::ApiKeys::default(),
            slot_save_path: None,
        };
        let mut slots = SlotFlags::default();
        let mut it = args.iter().map(String::as_str);
        while let Some(flag) = it.next() {
            if flag == "--help" || flag == "-h" {
                return Err(usage.into());
            }
            if flag == "--plan" {
                a.plan_only = true;
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {usage}"))?;
            match flag {
                "--host" => a.host = v.to_owned(),
                "--port" => a.port = serve::flag::number(flag, v)?,
                f if serve::flag::CTX.contains(&f) => a.ctx = Some(serve::flag::number(flag, v)?),
                "--place" => a.place = Some(place_word(v, None)?),
                "--alias" => a.alias = Some(v.to_owned()),
                TemplateFile::FLAG => a.template_file.set(v),
                "--queue-depth" => a.queue_depth = Some(serve::flag::number(flag, v)?),
                "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
                f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
                f if slots.take(f, v)? => {}
                other => {
                    if !own(other, v)? {
                        return Err(format!("unknown argument {other:?}: {usage}").into());
                    }
                }
            }
        }
        (a.ctx, a.parallel) = slots.given(a.ctx)?;
        if a.ctx == Some(0) {
            return Err("--ctx 0: the stores hold no position".into());
        }
        a.template_file.check()?;
        check_slot_ctx(a.ctx, a.parallel)?;
        Ok(a)
    }

    /// The common flags of `args` alone ([`SeatFlags::parse`]) for a seat whose `body` holds one
    /// sequence and serves no expert tier card: `--parallel` above one
    /// ([`SeatFlags::one_sequence`]) and a `--place` list with tier cards are refused by name.
    pub fn parse_one_sequence(
        args: &[String],
        usage: &str,
        body: &str,
    ) -> Result<SeatFlags, GateError> {
        let a = SeatFlags::parse(args, usage, |_, _| Ok(false))?;
        a.one_sequence(body)?;
        if let Some(place) = a.place {
            place.serves(body, 0)?;
        }
        Ok(a)
    }

    /// `--parallel` above one, refused by name for a seat whose `body` holds one sequence and no
    /// resident slots.
    pub fn one_sequence(&self, body: &str) -> Result<(), GateError> {
        match self.parallel.filter(|&n| n > 1) {
            Some(n) => Err(format!(
                "--parallel {n}: the {body} body holds one sequence and no resident slots; the \
                 seat serves --parallel 1"
            )
            .into()),
            None => Ok(()),
        }
    }

    /// [`bind_server`] over `engine` with these flags: the alias the flag
    /// gave or `chat`'s, `chat`'s template, no request preparation, the slot
    /// file path the flag gave, `slots` slots the seat made at its open, the
    /// server settings from `levers`, and the side model beside it.
    pub fn bind(
        self,
        engine: SeatEngine,
        chat: ChatSurface,
        path: &Path,
        slots: usize,
        levers: &Levers,
        side: Option<SideEngine>,
    ) -> Result<Server, ServeError> {
        let SeatFlags {
            host,
            port,
            alias,
            api_keys,
            queue_depth,
            slot_save_path,
            ..
        } = self;
        bind_server(
            engine,
            Listen {
                host: &host,
                port,
                alias: alias.unwrap_or(chat.name),
                path,
                template: chat.template,
                preparation: None,
                slot_save_path,
                api_keys,
                parallel: slots,
                queue_depth,
                schema_jump: levers.schema_jump(),
                side,
            },
        )
    }
}

/// The shape a seat's round of several slots runs in, decided once at the
/// open: the busy slots' rows as one pass of the body, or a select and a
/// step (a pass, drafted) a slot in turn ([`step_rows_in_turn`],
/// [`pass_rows_in_turn`]). The one owner of the `parallel` line's `pass=`
/// word; every seat passes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// One pass of the busy rows (or a load of one slot, which runs no round
    /// of several).
    One,
    /// A select and a step a slot.
    Turns,
}

impl Pass {
    /// The shape of a load of `slots` slots whose body runs a round of
    /// several as one pass when `one_pass`; one slot is [`Pass::One`]
    /// whatever the body does.
    pub fn of(slots: usize, one_pass: bool) -> Pass {
        if slots > 1 && !one_pass {
            Pass::Turns
        } else {
            Pass::One
        }
    }

    /// Whether a round of `slots` slots runs as one pass of the busy rows:
    /// several slots, and the shape [`Pass::One`].
    pub fn together(self, slots: usize) -> bool {
        slots > 1 && self == Pass::One
    }

    /// The `parallel` line's `pass=` word.
    pub fn word(self) -> &'static str {
        match self {
            Pass::One => "one",
            Pass::Turns => "turns",
        }
    }
}

/// The text of [`parallel_line`].
fn parallel_text(slots: usize, slot_ctx: usize, from: SlotsFrom, pass: Pass) -> String {
    format!(
        "parallel rule=slots slots={slots} slot_ctx={slot_ctx} total={} from={from} pass={}",
        slots * slot_ctx,
        pass.word()
    )
}

/// The `parallel` line a seat prints before its load: the slots, a slot's
/// context, the total, what set the count (`from`), and the shape of a round
/// of several slots (`pass`).
pub fn parallel_line(slots: usize, slot_ctx: usize, from: SlotsFrom, pass: Pass) {
    eprint_noted(&parallel_text(slots, slot_ctx, from, pass));
}

/// A seat's start line that is no [`crate::record::Record`] (the
/// `parallel`, `cache` and Qwen3.8 `load` lines) on stderr as it is, and to
/// the start screen (`serve::screen::note`), which keeps it only when armed.
pub fn eprint_noted(line: &str) {
    eprintln!("{line}");
    serve::screen::note(line);
}

/// Arms the start screen (`serve::screen`) when `on` (the seat's
/// `BLOOMERY_SCREEN`) and stderr is a terminal, and then hands it every
/// record line the seat prints; a seat calls it right after its levers and
/// records are registered, and the server draws the screen once it listens.
pub fn arm_screen(on: bool) {
    if serve::screen::arm(on) {
        crate::record::tee(serve::screen::note);
    }
}

/// A plain path has no drafted window whose width could be configured.
pub fn refuse_plain_width(
    levers: &bloomery_levers::Levers,
    why: Option<impl std::fmt::Display>,
) -> Result<(), GateError> {
    if let Some(why) = why
        && levers.mtp_width().is_some()
    {
        return Err(format!(
            "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; the server drafts nothing ({why})"
        ).into());
    }
    Ok(())
}

/// The arguments and optional lever reading shared by a seat's preflight and actual start, with
/// the lines of the levers set but not applied, which only the start prints ([`Self::run`]).
pub struct CheckedSeat<T> {
    args: T,
    levers: Option<bloomery_levers::Levers>,
    unapplied: Vec<String>,
}

impl<T> CheckedSeat<T> {
    /// The start's arguments and levers, once it prints the lines of the levers it does not apply.
    pub fn run(self) -> (T, bloomery_levers::Levers) {
        for line in &self.unapplied {
            eprintln!("{line}");
        }
        (self.args, self.levers.expect("the run reads its levers"))
    }
}

/// Reads a seat's levers, unless placement still has to supply them, and checks its arguments
/// through the same parser and model-independent rules its run uses. A check before the fetch
/// drops the result, so it prints no line for a lever another seat may apply.
pub fn check_seat<T>(
    acts_on: Option<&[&str]>,
    args: &[String],
    parse: impl FnOnce(&[String], Option<&bloomery_levers::Levers>) -> Result<T, GateError>,
) -> Result<CheckedSeat<T>, GateError> {
    let (levers, unapplied) = acts_on
        .map(|acts_on| {
            bloomery_levers::at_main_held(acts_on, |flag| {
                crate::llama_launch::serve_option(args, flag)
            })
        })
        .transpose()?
        .unzip();
    let parsed = parse(args, levers.as_ref())?;
    Ok(CheckedSeat {
        args: parsed,
        levers,
        unapplied: unapplied.unwrap_or_default(),
    })
}

/// `/props`' placement of `plan` on `machine`: its cards by nvidia-smi
/// index. A placement that cannot be named prints one line under `name` and
/// is left out.
pub fn placement_of(name: &str, machine: &Machine, plan: &Plan<'_>) -> Option<PlacementProps> {
    let placement = machine
        .all_cards()
        .map(|c| nvidia_smi_index(&c.name, c.device).map(|i| format!("GPU{i}")))
        .collect::<Result<Vec<String>, String>>()
        .and_then(|g| placement_props(plan, &g));
    if let Err(e) = &placement {
        eprintln!("{name}: /props leaves the placement out: {e}");
    }
    placement.ok()
}

/// A seat's [`Seat::note`]: a prefix kept less of than shared is a `cache
/// reuse` record; every other note prints as `name`'s line.
pub fn note_line(name: &str, note: &CacheNote) {
    if let CacheNote::Reuse {
        common,
        ask,
        kept,
        held,
        reason,
    } = note
    {
        Record::new(&crate::record::CACHE_REUSE)
            .u("common", common)
            .u("ask", ask)
            .u("kept", kept)
            .u("held", held)
            .w("reason", reason.as_deref().unwrap_or("unstated"))
            .eprint();
        return;
    }
    eprintln!("{name}: {note}");
}

/// The class toktape files a placement role's bytes under: the buckets of its
/// GGUF classifier, which puts the router and the shared expert with the
/// routed experts and every engram tensor with the n-gram tables.
fn class_of(role: Role) -> &'static str {
    match role {
        Role::Attention => "attention",
        Role::Router | Role::HashTable | Role::SharedExpert | Role::RoutedExperts => "experts",
        Role::FfnNorm | Role::DenseFfn => "ffn",
        Role::EngramDense | Role::EngramGain | Role::EngramTable | Role::EngramHash => "ngram",
        Role::TokenEmbedding => "embeddings",
        Role::Head => "output",
        Role::HyperConnection | Role::Unread | Role::Unused => "other",
    }
}

/// A stage's layers as `first-last`.
fn layer_range(layers: &Range<usize>) -> String {
    match layers.end.checked_sub(1) {
        Some(last) if last >= layers.start => format!("{}-{last}", layers.start),
        _ => "none".to_owned(),
    }
}

/// `/props`' `engine.model` of the file `split`, whose tensors its
/// architecture classified as `model`: the architecture, the type that holds
/// the most bytes of the tensors a step reads (the row-gathered tables — the
/// token embedding and the engram tables — and the never-read tensors left
/// out), the bytes of every shard on disk, the layer and expert counts, and the
/// training context. A fact the header does not state is left out.
pub fn model_props(split: &Split, model: &ModelTensors) -> ModelProps {
    let mut by_type: BTreeMap<&'static str, u64> = BTreeMap::new();
    for t in &model.tensors {
        let gathered_or_unread = matches!(
            t.role,
            Role::TokenEmbedding
                | Role::EngramTable
                | Role::EngramHash
                | Role::Unread
                | Role::Unused
        );
        if let (false, Some(name)) = (gathered_or_unread, t.ty.name()) {
            *by_type.entry(name).or_default() += t.file_bytes;
        }
    }
    let quant = by_type
        .iter()
        .max_by_key(|&(_, &bytes)| bytes)
        .map(|(&name, _)| name.to_owned());
    let bytes = (0..split.shard_count())
        .map(|i| {
            split
                .shard_path(i)
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
        })
        .sum();
    ModelProps {
        format: Some("gguf".to_owned()),
        arch: split.architecture().map(str::to_owned),
        quant,
        bytes,
        files: u64::try_from(split.shard_count()).ok(),
        n_layers: u64::try_from(model.layers).ok(),
        n_experts: Some(model.experts),
        n_experts_used: Some(model.experts_used),
        ctx_train: split.arch_get_u64("context_length"),
    }
}

/// `/props`' `engine.placement` of `plan`, the plan the engine loads by: per
/// card of the machine — the stage cards, then the expert tier cards — named
/// `gpus[c]` (`GPU<n>`, see [`nvidia_smi_index`]), the resident bytes of its
/// segments by class and a stage card's layers (a tier card runs none, and
/// its row has no `layers`); then the host's (`CPU`) the same way; and the
/// cards' KV bytes, cache and shadow. Only the plan's own rows are summed, so
/// a card's classes add up to its `dense_bytes + expert_bytes` and the host's
/// to its `expert_bytes + table_bytes`. The NVMe tier holds no resident bytes
/// and is no device here.
pub fn placement_props(plan: &Plan<'_>, gpus: &[String]) -> Result<PlacementProps, String> {
    let machine: Vec<_> = plan.machine.all_cards().collect();
    if gpus.len() != plan.cards.len() || machine.len() != plan.cards.len() {
        return Err(format!(
            "{} card names for a plan of {} cards on a machine of {}",
            gpus.len(),
            plan.cards.len(),
            machine.len()
        ));
    }
    let mut cards = vec![BTreeMap::<String, u64>::new(); plan.cards.len()];
    let mut host = BTreeMap::<String, u64>::new();
    for r in &plan.rows {
        let t = plan
            .model
            .tensors
            .get(r.tensor)
            .ok_or_else(|| format!("a plan row names tensor {}, past the model", r.tensor))?;
        for seg in &r.segments {
            let device = match seg.device {
                Device::Card(c) => cards.get_mut(c),
                Device::Host => Some(&mut host),
                Device::Nvme | Device::Unused => None,
            };
            if let Some(classes) = device {
                *classes.entry(class_of(t.role).to_owned()).or_default() += seg.resident_bytes;
            }
        }
    }
    let stages = plan.machine.cards.len();
    let mut devices: Vec<DeviceProps> = cards
        .into_iter()
        .zip(gpus)
        .zip(machine)
        .enumerate()
        .map(|(i, ((class_bytes, gpu), card))| DeviceProps {
            device: gpu.clone(),
            class_bytes,
            layers: (i < stages).then(|| layer_range(&card.layers)),
        })
        .collect();
    devices.push(DeviceProps {
        device: "CPU".to_owned(),
        class_bytes: host,
        layers: None,
    });
    Ok(PlacementProps {
        devices,
        vram_kv_bytes: Some(plan.cards.iter().map(|c| c.kv_bytes).sum()),
    })
}

/// The nvidia-smi index of the card a plan's card `name` (resolved to
/// `device` when the plan was, else found by `name`) opens on: the one
/// visible device of this process's census, matched to nvidia-smi's list by
/// its PCI bus id ([`workstation::listed_index`]).
/// Two cards of one name are two buses, so each plan card names its own
/// index; a card not in view, or a bus nvidia-smi does not list once, is an
/// error that says what was listed.
pub fn nvidia_smi_index(name: &str, device: Option<DeviceId>) -> Result<u32, String> {
    let census = bloomery_gpu::census().map_err(|e| format!("the device census: {e}"))?;
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi: {}", out.status));
    }
    workstation::listed_index(name, device, &census, &String::from_utf8_lossy(&out.stdout))
}

/// The most the prompt cache takes by default: llama-server's `--cache-ram`
/// default.
pub const CACHE_RAM_CAP: u64 = 8192 << 20;

/// The prompt cache's budget of a seat whose saved states live in host RAM
/// beside a load's host set (llama-server's `--cache-ram`, in MiB; 0 turns
/// it off), and its terms: set, as given; else the lesser of
/// [`CACHE_RAM_CAP`] and half of what the host's available bytes —
/// `MemAvailable`, or the smaller room under a cgroup v2 limit
/// (`workstation::host_available_read`) — leave past the plan's host need
/// and the residency's churn pool, the need holding the NVMe expert tier's
/// arena ([`CacheRam::of_tier`]) and the checkpoints the load's sequences
/// pin (`residency38::reserve_checkpoints`, the machine's host reserve), 0
/// when nothing is left — a zero the line's `why` names ([`CacheRam::why`]),
/// since a load with no prompt cache still serves correctly, every state
/// recomputed. A plan the NVMe expert tier split defaults to 0 too: the
/// split fills the room past the tier's floor with its arena and the host
/// experts it keeps, so what the reading leaves past the need is the split's
/// rounding or the reading's drift, never room the plan left. A default too
/// small for one saved state is 0 ([`CacheRam::holding`]).
pub struct CacheRam {
    pub ram: u64,
    /// `--cache-ram` given, or the default's terms.
    pub set: bool,
    /// The host's available bytes before the load.
    pub available: u64,
    /// Which reading gave `available` ([`HostRead`]).
    pub reading: HostRead,
    /// The plan's host need.
    pub need: u64,
    /// The residency's churn pool, 0 without one.
    pub pool: u64,
    /// The plan's NVMe expert tier arena, 0 without one: the anonymous
    /// bytes the tier's slots fill beside the plan's own host terms. Inside
    /// `need` already ([`HostNeed::bytes`] counts the arena); kept here for
    /// the line.
    pub tier: u64,
    /// The routed-expert bytes the plan's host leg reads from the NVMe
    /// tier (`HostTotals::nvme_expert_bytes`), 0 on a plan the tier did not
    /// split; above 0 the default budget is 0 (the type's doc).
    pub paged: u64,
    /// The checkpoint bytes the plan's machine reserves
    /// (`residency38::checkpoints_reserved`), 0 for a load that pins none.
    /// Inside `need` already; kept here for the line.
    pub checkpoints: u64,
    /// Why the default budget is 0, named for the line's `why=` term: the
    /// reading left nothing past the need and the pool; the NVMe tier's
    /// split took the room; or the budget was under one saved state. `None`
    /// for a set one (`--cache-ram 0` is the caller's own ask) and for a
    /// budget above 0, except one that holds no session at the context the
    /// load serves ([`CacheRam::short_of`]).
    pub why: Option<String>,
}

impl CacheRam {
    /// The `flag`'s value (`--cache-ram`, `--park-ram`), `mib` MiB, in
    /// bytes; refused by name past u64.
    pub fn parse_mib(flag: &str, mib: &str) -> Result<u64, GateError> {
        let n: u64 = serve::flag::number(flag, mib)?;
        Ok(n.checked_mul(1 << 20)
            .ok_or_else(|| format!("{flag} {n} MiB passes u64 bytes"))?)
    }

    /// The budget (the type's doc): `set` in bytes as given, else the
    /// default over `need` and `pool`, `need` holding the `checkpoints` the
    /// plan's machine reserves (`residency38::checkpoints_reserved`) —
    /// [`CacheRam::of_tier`] with no NVMe expert tier.
    pub fn of(
        set: Option<u64>,
        need: u64,
        pool: u64,
        checkpoints: u64,
    ) -> Result<CacheRam, GateError> {
        Self::of_tier(set, need, pool, (0, 0), checkpoints)
    }

    /// [`CacheRam::of`] on a plan the NVMe expert tier split: `(tier,
    /// paged)` its arena's bytes — inside `need` already ([`HostNeed::bytes`]
    /// counts the arena) — and the routed-expert bytes its host leg reads
    /// from the tier. A split plan's default is 0 (the type's doc): the
    /// arena's pages are anonymous, so a cache beside the room the split
    /// filled is the kernel's OOM or swap, not a page it reclaims.
    pub fn of_tier(
        set: Option<u64>,
        need: u64,
        pool: u64,
        (tier, paged): (u64, u64),
        checkpoints: u64,
    ) -> Result<CacheRam, GateError> {
        let (available, reading) = workstation::host_available_read()?;
        Self::at(
            set,
            (available, reading),
            need,
            pool,
            (tier, paged),
            checkpoints,
        )
    }

    /// [`CacheRam::of_tier`] at a host reading already taken: a caller that
    /// reads the budget at several needs (the default context's bound) takes
    /// one reading for all of them. A `need` under the `tier` and
    /// `checkpoints` it holds is refused by name: it is a need read without
    /// the arena or of a machine with no checkpoint reserve, and the budget
    /// would take their bytes.
    pub fn at(
        set: Option<u64>,
        (available, reading): (u64, HostRead),
        need: u64,
        pool: u64,
        (tier, paged): (u64, u64),
        checkpoints: u64,
    ) -> Result<CacheRam, GateError> {
        if tier > need {
            return Err(format!(
                "the prompt cache's need {need} B is under the NVMe tier arena {tier} B it holds: \
                 a need read without the arena (HostNeed::bytes counts it)"
            )
            .into());
        }
        if tier > 0 && paged == 0 {
            return Err(format!(
                "the prompt cache's plan holds an NVMe tier arena of {tier} B and pages no routed \
                 expert: the arena exists only on a plan the tier split"
            )
            .into());
        }
        if checkpoints > need - tier {
            return Err(format!(
                "the prompt cache's need {need} B is under the NVMe tier arena {tier} B and the \
                 checkpoints {checkpoints} B it holds: a need read of a machine with no \
                 checkpoint reserve (residency38::reserve_checkpoints)"
            )
            .into());
        }
        let (ram, why) = match set {
            Some(ram) => (ram, None),
            None if paged > 0 => (
                0,
                Some(format!(
                    "the NVMe tier's split fills the room past its floor: arena {tier} B and the \
                     host experts it keeps inside need {need} B, {paged} B read from the drive; \
                     no prompt cache, every state recomputed"
                )),
            ),
            None => Self::default_of(available, need, pool),
        };
        Ok(CacheRam {
            ram,
            set: set.is_some(),
            available,
            reading,
            need,
            pool,
            tier,
            paged,
            checkpoints,
            why,
        })
    }

    /// This budget, a default under `entry` bytes — the least one state the
    /// seat's cache saves holds (one position, its fixed terms whole) — made
    /// 0 with a `why` naming both: the cache would refuse every state it is
    /// handed (`serve`'s `PromptCache::insert`), so it is off. A set budget
    /// keeps its value: it is the caller's ask, and the seat refuses one too
    /// small by name where its rule reads it.
    #[must_use]
    pub fn holding(self, entry: u64) -> CacheRam {
        if self.set || self.ram == 0 || self.ram >= entry {
            return self;
        }
        CacheRam {
            ram: 0,
            why: Some(format!(
                "the default budget {} B is under one saved state's {entry} B: no prompt cache, \
                 every state recomputed",
                self.ram
            )),
            ..self
        }
    }

    /// This budget, one sequence state of `state` bytes at the `ctx` positions
    /// a slot serves over it: kept, with a `why` naming both. The default
    /// context never falls under the base for the budget's sake, and the cache
    /// refuses a state over its whole budget and keeps the shorter ones it
    /// holds, so a budget that holds no session at the base stays on for the
    /// sessions it holds. A budget of 0 has its own `why`.
    #[must_use]
    pub fn short_of(self, ctx: usize, state: u64) -> CacheRam {
        if self.ram == 0 || self.ram >= state {
            return self;
        }
        CacheRam {
            why: Some(format!(
                "the budget {} B is under one session's state of {state} B at the {ctx} positions \
                 a slot serves: the cache keeps the shorter sessions it holds and refuses the \
                 longer ones",
                self.ram
            )),
            ..self
        }
    }

    /// The default budget ([`CacheRam`]'s doc): half of what `available`
    /// leaves past `need` and `pool`, capped at [`CACHE_RAM_CAP`]. A
    /// leftover of nothing is a named record, not a silent zero: the seat
    /// serves correctly without a prompt cache (every state recomputed), so
    /// the line states the arithmetic that left nothing instead of the load
    /// refusing.
    fn default_of(available: u64, need: u64, pool: u64) -> (u64, Option<String>) {
        let left = i128::from(available) - i128::from(need) - i128::from(pool);
        if left / 2 <= 0 {
            return (
                0,
                Some(format!(
                    "the reading {available} B leaves {left} B past need {need} B + pool {pool} B: \
                     no prompt cache, every state recomputed"
                )),
            );
        }
        (
            u64::try_from(left / 2).map_or(CACHE_RAM_CAP, |h| h.min(CACHE_RAM_CAP)),
            None,
        )
    }

    /// The `cache` line's terms, the seat's own fields to follow: `cache
    /// ram=… rule=set|default available=… read=… need=… pool=… tier=…
    /// paged=… checkpoints=…`, `read` the reading that gave `available`
    /// ([`HostRead::word`]); a default budget of 0 carries `why=…`
    /// ([`CacheRam::why`]).
    #[must_use]
    pub fn line(&self) -> String {
        let line = format!(
            "cache ram={} rule={} available={} read={} need={} pool={} tier={} paged={} \
             checkpoints={}",
            self.ram,
            if self.set { "set" } else { "default" },
            self.available,
            self.reading.word(),
            self.need,
            self.pool,
            self.tier,
            self.paged,
            self.checkpoints
        );
        match &self.why {
            Some(why) => format!("{line} why={why}"),
            None => line,
        }
    }
}

/// The bytes of one chunk of a slot file's pipe (the module doc).
const PIPE_CHUNK: usize = 4 << 20;

/// The chunks a slot file's pipe holds waiting: with the one each thread
/// holds in hand, the host peak past the seat's own copy is
/// `(PIPE_DEPTH + 2) · PIPE_CHUNK`.
const PIPE_DEPTH: usize = 2;

/// What the engine thread is asked to do.
enum Cmd {
    Prefill(Vec<u32>),
    /// [`Seat::prefill_media`]: `ids` with the images they carry, one feed a
    /// span, every span whole inside the call ([`Engine::prefill_media`]'s
    /// contract).
    PrefillMedia(Vec<u32>, Vec<MediaFeed>),
    /// With the engine's logits buffer when the caller wants the row.
    Next {
        last: u32,
        logits: Option<Vec<f32>>,
    },
    /// `ids` and the step of the last ([`Seat::prefill_row`]), the logits
    /// buffer lent as a `Next`'s.
    NextAfter {
        ids: Vec<u32>,
        logits: Option<Vec<f32>>,
    },
    /// A prompt's last call and the step of its last id
    /// ([`Seat::prefill_next`]), the logits buffer lent as a `Next`'s.
    PrefillNext {
        ids: Vec<u32>,
        logits: Option<Vec<f32>>,
    },
    /// A pass from `last`, its kept tokens into the caller's buffer.
    Pass {
        last: u32,
        out: Vec<u32>,
    },
    /// A sampled pass from `last` ([`Seat::pass_sampled`]): the request's
    /// ids so far and its sampler cross to the thread and back with the
    /// reply, the sampler's state with it.
    PassSampled {
        last: u32,
        history: Vec<u32>,
        sampler: serve::Sampler,
        best: Option<usize>,
        out: Vec<u32>,
    },
    Reset,
    /// Take back the positions from this one on.
    Rollback(u32),
    /// The longest prefix of at most this many positions a rollback keeps
    /// for a reply of at most this many tokens through passes
    /// ([`Seat::keep_for`]), and the rule that kept less.
    Keep(usize, Option<usize>),
    /// Where to cut a prompt call of `first .. end` at `marks`.
    Splits {
        first: usize,
        end: usize,
        marks: Vec<usize>,
    },
    /// The sequence state, as a value.
    Save,
    SaveSize,
    /// The sequence state, as a value whose copy may still run
    /// ([`Seat::snapshot_begin`]).
    SaveBegin,
    /// What a begun copy cost if it has landed, `None` if it runs
    /// ([`serve::Flight::poll`]), asked on the engine thread.
    SavePoll(Pending),
    /// The wait for a begun copy ([`serve::Flight::wait`]), on the engine
    /// thread.
    SaveWait(Pending),
    /// Replace the sequence state with a saved one.
    Resume(Arc<dyn Saved>),
    /// Replace the sequence state with the first positions of a saved one
    /// ([`Seat::resume_prefix`]).
    ResumePrefix(Arc<dyn Saved>, usize),
    /// What the selected slot can take of this many positions of that slot
    /// ([`Seat::share_keep`]).
    ShareKeep(usize, usize),
    /// Give the selected slot the first positions of that slot, at most this
    /// many ([`Seat::share_prefix`]).
    Share(usize, usize),
    /// The selected slot's state written into the pipe ([`save_into`]).
    SaveState(SyncSender<Vec<u8>>),
    /// The selected slot's state read from the pipe ([`restore_from`]).
    RestoreState(Receiver<Vec<u8>>),
    /// Nothing: the reply carries the position the model stands at.
    Pos,
    /// The residency back to its seed.
    ResidencyReset,
    /// The seat's last command ([`Seat::close`]).
    Close,
    /// Make `slot` the one every later command acts on ([`Seat::select`]).
    Select(usize),
    /// A step of several slots in one command ([`SeatEngine`]'s
    /// `step_slots`): each row's `last` on its own slot, in row order, the
    /// answers and lent rows back with the one reply.
    StepSlots(Vec<SlotStep>),
    /// A drafted pass of several slots in one command ([`SeatEngine`]'s
    /// `advance_slots`): each row's `last` passed on its own slot, in row
    /// order, the kept ids and what each pass drafted back with the one
    /// reply.
    PassSlots(Vec<SlotPassRow>),
}

/// One row of a [`Cmd::StepSlots`] and of a [`Seat::step_slots`] round: the
/// slot, the id it evaluates, the engine's logits row lent for it
/// (`n_vocab` f32, filled only when the row asked), and the engine's
/// answer, which the seat sets.
pub struct SlotStep {
    pub slot: usize,
    pub last: u32,
    pub logits: Option<Vec<f32>>,
    pub next: u32,
}

/// One row of a [`Cmd::PassSlots`] and of a [`Seat::pass_slots`] round: the
/// slot, the id its pass runs, the caller's buffer of the pass's kept ids,
/// what the pass drafted, which the seat sets, and the draw of a request
/// that draws (`serve::SlotDraw`), whose pass is [`Seat::pass_sampled`]'s.
pub struct SlotPassRow {
    pub slot: usize,
    pub last: u32,
    pub out: Vec<u32>,
    pub drafted: Drafted,
    pub draw: Option<RowDraw>,
}

/// What a sampled row of a [`Cmd::PassSlots`] draws with, crossing to the
/// engine thread and back with the row: the request's history, handed back
/// as it was given, its sampler, its state moved one draw a kept id, and the
/// most of a row's best logits the sampler reads (`serve::SlotDraw`'s
/// `best`) ([`Seat::pass_sampled`]'s).
pub struct RowDraw {
    pub history: Vec<u32>,
    pub sampler: serve::Sampler,
    pub best: Option<usize>,
}

/// What a reply carries besides its result.
enum Extra {
    None,
    /// A `Keep`'s rule.
    Why(Option<String>),
    /// A `Splits`' cuts.
    Splits(Vec<usize>),
    /// A `Save`'s state, or its failure with its kind ([`snapshot_of`]).
    Saved(Result<Arc<dyn Saved>, StateError>),
    SaveSize(Result<Option<u64>, StateError>),
    /// A `SaveBegin`'s state with its copy in flight, or its failure with its
    /// kind ([`begin_of`]).
    Begun(Result<Pending, StateError>),
    /// A `SavePoll`'s or `SaveWait`'s answer: what the copy cost once it has
    /// landed, `None` while it runs (a wait never answers `None`).
    Polled(Result<Option<Landed>, StateError>),
    /// A `Resume`'s outcome with its kind ([`resume_of`]).
    Resumed(Result<(), StateError>),
    /// A `ResumePrefix`' positions put back, or its failure with its kind
    /// ([`resume_prefix_of`]).
    Prefix(Result<usize, StateError>),
    /// A `Share`'s outcome, or its failure with its kind ([`share_of`]).
    Shared(Result<SharedPrefix, StateError>),
    /// A `SaveState`'s or `RestoreState`'s outcome, its kind as the seat
    /// gave it.
    State(Result<SavedState, StateError>),
    /// A `Pass`'s buffer, holding its kept tokens on success, and what its
    /// draft proposed and kept.
    Pass(Vec<u32>, Drafted),
    /// A `PassSampled`'s buffer, holding its taken ids on success, what its
    /// draft proposed and kept, and the history and sampler it was sent.
    PassSampled {
        out: Vec<u32>,
        drafted: Drafted,
        history: Vec<u32>,
        sampler: serve::Sampler,
    },
    /// A `ResidencyReset`'s report; `None` for a seat with no residency.
    Residency(Option<ResidencyReset>),
    /// A `StepSlots`' rows, their answers set and their lent rows filled.
    StepSlots(Vec<SlotStep>),
    /// A `PassSlots`' rows, their kept ids and drafted counts set.
    PassSlots(Vec<SlotPassRow>),
}

/// Its answer: the argmax of a `Next` (the kept length of a `Keep`), the
/// logits buffer a `Next` was lent (filled unless the result is an error),
/// the position it stands at afterwards (the position it failed at, on an
/// error), and what a `Keep` or `Splits` returns besides. A `Save`,
/// `Resume`, `SaveState` or `RestoreState` answers in its extra alone, with
/// its kind.
struct Reply {
    result: Result<u32, String>,
    logits: Option<Vec<f32>>,
    pos: usize,
    extra: Extra,
}

/// What the engine thread runs every command on: the model standing at a
/// position, the draft that follows it if one runs, and the body's rules
/// ([`SeatEngine::spawn`] opens it on the thread). A failed call says what
/// failed; the thread adds the command and the position.
pub trait Seat: 'static {
    /// The position the next fed id lands in.
    fn pos(&self) -> usize;
    /// The positions the caches were sized for: `pos` never passes it.
    fn ctx_max(&self) -> usize;
    /// The body's prompt feed of `ids` from `pos` (a batched body's batch, or
    /// one step per id), the draft fed as the call hands its rows over; the
    /// argmax after the last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError>;
    /// Prompt positions whose output is unused. A batched seat can omit its head.
    fn prefill_only(&mut self, ids: &[u32]) -> Result<(), GateError> {
        self.prefill(ids).map(|_| ())
    }
    /// [`Seat::prefill`] where the slice carries images
    /// ([`Engine::prefill_media`]'s contract): each feed's `at` is its span's
    /// first position in `ids`, every span lies whole inside the slice, and
    /// every position of a span carries the model's image token. The default
    /// is [`Seat::prefill`] when the slice carries no image, and a named
    /// refusal when it does.
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        if feeds.is_empty() {
            return self.prefill(ids);
        }
        Err("this seat takes no image input".into())
    }
    /// The seat's image input, `None` for a seat that takes none
    /// ([`Engine::media_model`]): read once, at the load. The default is
    /// `None`.
    fn media_model(&self) -> Option<SharedMediaModel> {
        None
    }
    /// One step on `last` at `pos`, the draft fed its row; the argmax after it.
    fn step(&mut self, last: u32) -> Result<u32, GateError>;
    /// The head's logits after the last step into `row` (`n_vocab` f32).
    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError>;
    /// One step on `last` at `pos` ([`Seat::step`]) and, into `row`, the
    /// target's logits of that step ([`Seat::logits_into`]). A seat whose step
    /// runs a draft after the target's step overrides it to read the row
    /// between the two, so the row is the target's whatever the draft writes.
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let arg = self.step(last)?;
        self.logits_into(row)
            .map_err(|e| format!("logits of the step: {e}"))?;
        Ok(arg)
    }
    /// `ids` (at least one) from `pos` and the step of the last
    /// (`serve::Engine::next_after`): the argmax after the last and, into
    /// `row`, that position's logits. The default is [`Seat::prefill`] of all
    /// but the last id and [`Seat::step`] (or [`Seat::step_row`]) of the
    /// last; a seat whose prefill already ran the head on its last row
    /// overrides it to read that row.
    fn prefill_row(&mut self, ids: &[u32], row: Option<&mut [f32]>) -> Result<u32, GateError> {
        prefill_then_step(self, ids, row)
    }
    /// A prompt's last call `ids` (at least one) from the slot's position and
    /// the step of its last id (`serve::Engine::prefill_next`): the argmax after the last
    /// and, into `row`, that position's logits. The prompt call takes its
    /// checkpoints as [`Seat::prefill_only`] of all but the last id does. The
    /// default is that call and [`Seat::step`] (or [`Seat::step_row`]) of the
    /// last; a seat whose prompt call runs the last id as its last row
    /// overrides it.
    fn prefill_next(&mut self, ids: &[u32], row: Option<&mut [f32]>) -> Result<u32, GateError> {
        prefill_then_step(self, ids, row)
    }
    /// The position grid on which the seat's prompt calls are bit-neutral to
    /// cut (`serve::Engine::prompt_quantum`), read once at the load. The
    /// default is `None`: every prompt call runs in one go.
    fn prompt_quantum(&self) -> Option<NonZeroUsize> {
        None
    }
    /// One pass from `last` (`serve::Engine::advance`): the kept tokens into
    /// `out`, and what the draft proposed and kept. Without a draft, one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        out.push(self.step(last)?);
        Ok(Drafted::default())
    }
    /// The most positions [`Seat::pass`] runs.
    fn pass_rows(&self) -> usize {
        1
    }
    /// One sampled pass from `last` (`serve::Engine::advance_sampled`): the
    /// draft's proposal verified, each row's id drawn by `sampler` from that
    /// row's logits given `history` and the ids this pass took before it,
    /// up to the first that is not the proposal's next; the taken ids into
    /// `out`, and what the draft proposed and kept. `history` comes back as
    /// it was given, the sampler with its state moved one draw a taken id.
    /// `best` is the most of a row's best logits the sampler reads
    /// (`serve::Engine::advance_sampled`'s). Asked only of a seat that
    /// [`Seat::drafts_sampled`]; the default refuses by name.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut serve::Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let _ = (last, history, sampler, best, out);
        Err("this seat does not draft a sampled request".into())
    }
    /// Whether the seat runs [`Seat::pass_sampled`]: it drafts and reads its
    /// verify rows' logits (`serve::Engine::drafts_sampled`). The default is
    /// false.
    fn drafts_sampled(&self) -> bool {
        false
    }
    /// The sequences this seat serves at once, each a sequence of its own
    /// the seat keeps resident (the server steps them together, one
    /// [`Seat::step`] each in one round). The default is 1: the seat serves
    /// one sequence, and several slots take it in turns (`serve::SwapEngine`).
    fn slots(&self) -> usize {
        1
    }
    /// Make `slot` (below [`Seat::slots`]) the one every later call acts on
    /// until the next select, exchanging the seat's resident sequences. The
    /// default serves slot 0 alone: selecting it is a no-op, any other slot
    /// refused by name.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if slot == 0 {
            Ok(())
        } else {
            Err(format!("slot {slot}: this seat serves slot 0 alone").into())
        }
    }
    /// Whether the seat keeps its draft's state per slot — parked on a
    /// [`Seat::select`], put back with the slot — so a slot's drafted passes
    /// are the passes it would run alone whatever the other slots run
    /// (`serve::Engine::slot_drafts`); the server serves several slots of a
    /// drafting seat only when it declares this. A seat that drafts nothing
    /// needs no declaration. The default is false.
    fn slot_drafts(&self) -> bool {
        false
    }
    /// Whether the seat counts its rounds of several slots — a `slots round`
    /// record a round (`record::slots_round`), and while it serves several
    /// a `slot call` record a call of one (`record::slot_call`, the engine
    /// thread's) — under `BLOOMERY_STEP_STATS`: the lever's value the
    /// binary's `main` parsed, carried by a seat whose binary names the
    /// lever among those it acts on. The default counts
    /// nothing: a binary that does not name the lever refuses it set, so its
    /// seats never count.
    fn step_stats(&self) -> bool {
        false
    }
    /// One round of several slots' steps (`serve::Engine::step_slots` through
    /// [`SeatEngine`]): each row's `last` stepped on its own slot, in row
    /// order, the answers written to `next` and each lent logits row filled.
    /// A row that fails ends the round there; the error is the server's to
    /// die on, and the rows cross back with the reply whole whatever the
    /// result, their lent rows included.
    ///
    /// The default is the fallback for a seat with no one-pass override
    /// ([`step_rows_in_turn`]: a select and a step a row). The seats that
    /// override it, or will: the qwen3moe whole-card seat (`serve_seats::qwen3`,
    /// one pass of the busy rows over the session); Qwen3.6 (round O3b, once
    /// its body serves resident slots); V4.1 (round T3 `stgseat`); GLM
    /// (line3's `glmseatpass`).
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        step_rows_in_turn(self, rows)
    }
    /// One round of several slots' drafted passes (`serve::Engine::advance_slots`
    /// through [`SeatEngine`]): each row's `last` passed on its own slot, in
    /// row order, the kept ids into `out` and what the draft proposed and
    /// kept into `drafted`. A row that fails ends the round there; the error
    /// is the server's to die on.
    ///
    /// The default is the fallback for a seat with no one-pass override
    /// ([`pass_rows_in_turn`]: a select and a pass a row). The seats that
    /// override it, or will: the Qwen3.8 drafted seat (round O3c, through
    /// `app::mtp::pass_slots`); V4.1 (round T3 `stgseat`); GLM (line3's
    /// `glmseatpass`).
    fn pass_slots(&mut self, rows: &mut [SlotPassRow]) -> Result<(), String> {
        pass_rows_in_turn(self, rows)
    }
    /// Empty caches at position 0.
    fn reset(&mut self) -> Result<(), GateError>;
    /// Take back the positions from `pos` on; `pos` is one [`Seat::keep`]
    /// granted.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError>;
    /// The longest prefix of at most `n` positions (no more than the model
    /// holds) a rollback keeps and the next request can run from, and the
    /// rule that kept less.
    fn keep(&self, n: usize) -> (usize, Option<String>);
    /// [`Seat::keep`] for a request whose reply makes at most `reply` tokens
    /// through passes (`serve::Engine::will_reply`): a seat whose kept prefix
    /// costs its draft weighs one against the other. The default is `keep`'s.
    fn keep_for(&self, n: usize, reply: Option<usize>) -> (usize, Option<String>) {
        let _ = reply;
        self.keep(n)
    }
    /// Where to cut a prompt call of `first .. end` (from where the model
    /// stands) so the `marks` stay keepable (`serve::Engine::prefill_splits`).
    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize>;
    /// The sequence state as a value the server's prompt cache holds. A
    /// refusal by name — a `bloomery_gpu::GpuError` of kind `State` (the
    /// model does not hold what the call needs) or `Shape` (a call or a
    /// state it does not take), anywhere in the error's chain — is the
    /// request's: the server keeps no state and serves on. Any other error is
    /// the engine's, which ends the server.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError>;
    /// The host array bytes of a snapshot, without reading its stores.
    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        Ok(None)
    }
    /// Whether [`Seat::snapshot_begin`] copies the state on a stream of the
    /// seat's own, so the server may ask for it when a request ends without
    /// stopping the slots (`serve::Engine::snapshots_async`). The default is
    /// false: the snapshot is taken on the request path.
    fn snapshots_async(&self) -> bool {
        false
    }
    /// [`Seat::snapshot`] whose copy may still run when the call returns:
    /// the state's positions, bytes and rules are known at once, its bytes
    /// are read only after the [`Pending`]'s flight has landed, and the
    /// slot's rows are not cut, reset or replaced before then
    /// (`serve::Engine::snapshot_begin`). The default is the snapshot, landed
    /// already.
    fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
        Ok(Pending::landed(self.snapshot()?))
    }
    /// Replace the sequence state with `state`, which [`Seat::snapshot`] took
    /// of this seat. A refusal by name ([`Seat::snapshot`]'s kinds) is the
    /// request's: the server drops the state and resets the slot.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError>;
    /// Replace the sequence state with the first `n` positions of `state`
    /// (`serve::Engine::resume_prefix`): only the rows and the stores of those
    /// positions go to the card; returns the positions the seat holds after.
    /// The default is [`Seat::resume`] of the whole state.
    fn resume_prefix(&mut self, state: &dyn Saved, n: usize) -> Result<usize, GateError> {
        let _ = n;
        self.resume(state)?;
        Ok(state.n_tokens())
    }
    /// The most of the first `n` positions slot `from` holds that the
    /// selected slot can take from it (`serve::Engine::share_keepable`). The
    /// default is 0: the seat shares nothing between its slots, and
    /// [`Seat::share_prefix`] is never asked.
    fn share_keep(&self, from: usize, n: usize) -> usize {
        let _ = (from, n);
        0
    }
    /// The selected slot given the first positions slot `from` holds, at
    /// most `n`, as [`Seat::share_keep`] granted
    /// (`serve::Engine::share_prefix`). A refusal by name
    /// ([`Seat::snapshot`]'s kinds) is the request's: the server resets the
    /// slot. The default refuses by name.
    fn share_prefix(&mut self, from: usize, n: usize) -> Result<SharedPrefix, GateError> {
        Err(format!("this seat shares no prefix between slots (slot {from}, {n} positions)").into())
    }
    /// The selected slot's sequence state written to `out` in a form
    /// [`Seat::restore_state`] reads back (`serve::Engine::save_state`), the
    /// state unchanged; the bytes written. A refusal is `Format`, an error
    /// of `out` is the stream's (`Io`), never the model's; only a failure
    /// that leaves the model in no defined state is `Engine`, which ends the
    /// server. The default writes nothing and refuses with `Unsupported`: the
    /// server answers 501. The seats that override it: V4.1
    /// (`serve_seats::ds41`) and Qwen3.8 (`serve_seats::qwen38`, its state
    /// through `SeqState`'s byte form); GLM once its sequence state has one.
    fn save_state(&mut self, out: &mut dyn Write) -> Result<u64, StateError> {
        let _ = out;
        Err(StateError::Unsupported("slot save/restore"))
    }
    /// The selected slot's sequence state replaced with the one `input`
    /// carries, which runs to its end (`serve::Engine::restore_state`); the
    /// positions restored, where the seat then stands. The kinds are
    /// [`Seat::save_state`]'s: a refusal after the seat has started to read
    /// leaves the slot to the server, which resets it. The default reads
    /// nothing and refuses with `Unsupported`; the seats that override it are
    /// [`Seat::save_state`]'s.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<usize, StateError> {
        let _ = input;
        Err(StateError::Unsupported("slot save/restore"))
    }
    /// `/props`' `engine` object: `p`, the opener's, with what only the load
    /// learned (a draft's device and resident bytes). The default adds
    /// nothing.
    fn props(&self, p: EngineProps) -> EngineProps {
        p
    }
    /// The residency back to its seed, the sequence where it stands, and
    /// its report (the seat prints its record); `None`, the default, for a
    /// seat with no residency.
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, GateError> {
        Ok(None)
    }
    /// The server's last call (`serve::Engine::close`), nothing freed: the
    /// records the seat still holds of its slots' last requests printed (a
    /// draft's `mtp width`), then the load ended on every expert tier's
    /// node. The default, for a seat that holds no record and whose model
    /// runs no tier on a node, does nothing.
    fn close(&mut self) -> Result<(), GateError> {
        Ok(())
    }
    /// Prints what the server's prompt cache did, as the binary's records.
    fn note(note: &CacheNote)
    where
        Self: Sized;
}

/// `serve::Engine` over a [`Seat`] that lives on its own thread. The
/// position is the model's alone: every question about it goes to the thread.
pub struct SeatEngine {
    link: Link,
    worker: Option<JoinHandle<()>>,
    vocab: Arc<Vocab>,
    ctx_max: usize,
    /// [`Seat::pass_rows`].
    rows: usize,
    /// [`Seat::slots`], the opened seat's.
    slots: usize,
    /// [`Seat::slot_drafts`], the opened seat's.
    slot_drafts: bool,
    /// [`Seat::drafts_sampled`], the opened seat's.
    drafts_sampled: bool,
    /// [`Seat::snapshots_async`], the opened seat's.
    snapshots_async: bool,
    /// [`Seat::media_model`], the opened seat's.
    media: Option<SharedMediaModel>,
    /// [`Seat::prompt_quantum`], the opened seat's.
    quantum: Option<NonZeroUsize>,
    card: String,
    props: EngineProps,
    cache_ram: u64,
    note: fn(&CacheNote),
    /// The request's reply (`serve::Engine::will_reply`), which each keep
    /// query carries.
    reply: Option<usize>,
}

/// The handle's side of the engine thread: its channels and the logits buffer
/// a sampled step borrows. It keeps no position; the thread's replies carry it.
struct Link {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Reply>,
    /// The logits row a sampled `next` reads (`n_vocab` f32), lent to the
    /// engine thread for the call; empty while lent or when it never came back.
    logits: Vec<f32>,
    /// The logits rows a step of several slots reads, one a row of the round
    /// (`n_vocab` f32, grown once and reused every round): lent with the
    /// command, filled on the engine thread, handed back with the reply.
    scratch: Vec<Vec<f32>>,
}

impl SeatEngine {
    /// Start the engine thread, open the seat on it with `open` (which
    /// writes the load lines), and wait until it is loaded. `defined` is how
    /// many positions the body computes the model at; the engine serves the
    /// lesser of it and the cache. `card` names the device in a crash report;
    /// `props` is what `/props` reports about the engine ([`model_props`],
    /// [`placement_props`]), which the seat completes ([`Seat::props`]);
    /// `cache_ram` is the server's prompt cache budget
    /// (`serve::Engine::cache_ram`). The seat's whole-load capabilities —
    /// [`Seat::pass_rows`], [`Seat::slots`], [`Seat::slot_drafts`],
    /// [`Seat::drafts_sampled`] — cross with the load.
    pub fn spawn<S, F>(
        open: F,
        defined: usize,
        vocab: Arc<Vocab>,
        card: String,
        props: EngineProps,
        cache_ram: u64,
    ) -> Result<SeatEngine, GateError>
    where
        S: Seat,
        F: FnOnce() -> Result<S, GateError> + Send + 'static,
    {
        crate::record::require(crate::record::groups::bind::CACHE, "bind::spawn");
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (replies, rx) = mpsc::channel::<Reply>();
        let (opened, loaded) = mpsc::channel::<
            Result<
                (
                    usize,
                    usize,
                    usize,
                    bool,
                    bool,
                    bool,
                    bool,
                    Option<SharedMediaModel>,
                    Option<NonZeroUsize>,
                    EngineProps,
                ),
                String,
            >,
        >();
        let n_vocab = vocab.n_vocab();
        let worker = std::thread::Builder::new()
            .name("engine".to_owned())
            .spawn(move || {
                let mut g = match open() {
                    Ok(g) => g,
                    Err(e) => {
                        let _ = opened.send(Err(e.to_string()));
                        return;
                    }
                };
                if opened
                    .send(Ok((
                        g.ctx_max(),
                        g.pass_rows(),
                        g.slots(),
                        g.slot_drafts(),
                        g.drafts_sampled(),
                        g.snapshots_async(),
                        g.step_stats() && g.slots() >= 2,
                        g.media_model(),
                        g.prompt_quantum(),
                        g.props(props),
                    )))
                    .is_err()
                {
                    return;
                }
                // The slot the server's last select named: slot 0 from the
                // start, none after a round of several, which leaves the
                // seat's selection unspecified until the server selects again
                // (`serve::Engine::step_slots`).
                let mut selected = Some(0);
                while let Ok(cmd) = receive_command(&cmds) {
                    let call = one_slot_call(&cmd);
                    let select = match &cmd {
                        Cmd::Select(slot) => Some(*slot),
                        _ => None,
                    };
                    let round = matches!(cmd, Cmd::StepSlots(_) | Cmd::PassSlots(_));
                    let (mut result, logits, extra) = match cmd {
                        Cmd::Keep(n, reply) => {
                            let (k, why) = g.keep_for(n.min(g.pos()), reply);
                            (
                                u32::try_from(k).map_err(|_| {
                                    format!("a kept prefix of at most {n} passes u32")
                                }),
                                None,
                                Extra::Why(why),
                            )
                        }
                        Cmd::Splits { first, end, marks } => {
                            (Ok(0), None, Extra::Splits(g.splits(first, end, &marks)))
                        }
                        Cmd::Pass { last, mut out } => {
                            let at = g.pos();
                            out.clear();
                            match pass(&mut g, last, &mut out) {
                                Ok(d) => (Ok(0), None, Extra::Pass(out, d)),
                                Err(e) => (
                                    Err(format!("pass at position {at}: {e}")),
                                    None,
                                    Extra::Pass(out, Drafted::default()),
                                ),
                            }
                        }
                        Cmd::PassSampled {
                            last,
                            mut history,
                            mut sampler,
                            best,
                            mut out,
                        } => {
                            let at = g.pos();
                            out.clear();
                            let r = pass_sampled(
                                &mut g,
                                last,
                                &mut history,
                                &mut sampler,
                                best,
                                &mut out,
                            );
                            let (result, drafted) = match r {
                                Ok(d) => (Ok(0), d),
                                Err(e) => (
                                    Err(format!("sampled pass at position {at}: {e}")),
                                    Drafted::default(),
                                ),
                            };
                            let extra = Extra::PassSampled {
                                out,
                                drafted,
                                history,
                                sampler,
                            };
                            (result, None, extra)
                        }
                        Cmd::Save => (Ok(0), None, Extra::Saved(snapshot_of(&mut g))),
                        Cmd::SaveSize => (Ok(0), None, Extra::SaveSize(snapshot_size_of(&mut g))),
                        Cmd::SaveBegin => (Ok(0), None, Extra::Begun(begin_of(&mut g))),
                        Cmd::SavePoll(p) => (Ok(0), None, Extra::Polled(p.flight().poll())),
                        Cmd::SaveWait(p) => {
                            (Ok(0), None, Extra::Polled(p.flight().wait().map(Some)))
                        }
                        Cmd::ResidencyReset => match g.residency_reset() {
                            Ok(r) => (Ok(0), None, Extra::Residency(r)),
                            Err(e) => (
                                Err(format!("residency reset at position {}: {e}", g.pos())),
                                None,
                                Extra::None,
                            ),
                        },
                        Cmd::Close => (
                            g.close()
                                .map(|()| 0)
                                .map_err(|e| format!("closing the seat: {e}")),
                            None,
                            Extra::None,
                        ),
                        Cmd::Resume(state) => {
                            (Ok(0), None, Extra::Resumed(resume_of(&mut g, &*state)))
                        }
                        Cmd::ResumePrefix(state, n) => (
                            Ok(0),
                            None,
                            Extra::Prefix(resume_prefix_of(&mut g, &*state, n)),
                        ),
                        Cmd::ShareKeep(from, n) => (
                            u32::try_from(g.share_keep(from, n))
                                .map_err(|_| format!("a shared prefix of at most {n} passes u32")),
                            None,
                            Extra::None,
                        ),
                        Cmd::Share(from, n) => {
                            (Ok(0), None, Extra::Shared(share_of(&mut g, from, n)))
                        }
                        Cmd::SaveState(chunks) => {
                            (Ok(0), None, Extra::State(save_into(&mut g, chunks)))
                        }
                        Cmd::RestoreState(chunks) => {
                            (Ok(0), None, Extra::State(restore_from(&mut g, chunks)))
                        }
                        Cmd::StepSlots(mut rows) => {
                            // The seat's round ([`Seat::step_slots`]): one
                            // command on the thread, its rows back with the
                            // one reply — not two round trips a row.
                            let r = g.step_slots(&mut rows).map(|()| 0);
                            (r, None, Extra::StepSlots(rows))
                        }
                        Cmd::PassSlots(mut rows) => {
                            // The seat's round ([`Seat::pass_slots`]): one
                            // command on the thread, its rows back with the
                            // one reply — not two round trips a row.
                            let r = g.pass_slots(&mut rows).map(|()| 0);
                            (r, None, Extra::PassSlots(rows))
                        }
                        cmd => {
                            let (result, logits) = serve_cmd(&mut g, cmd, n_vocab);
                            (result, logits, Extra::None)
                        }
                    };
                    if let Some(slot) = select {
                        selected = result.is_ok().then_some(slot);
                    }
                    if round {
                        selected = None;
                    }
                    if let Some(call) = call
                        && result.is_ok()
                        && let Err(e) = count_call(&g, call, selected)
                    {
                        result = Err(e);
                    }
                    if replies
                        .send(Reply {
                            result,
                            logits,
                            pos: g.pos(),
                            extra,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })?;
        let (ctx_max, rows, slots, drafts, media, quantum, props, counts) = match loaded.recv() {
            Ok(Ok((c, rows, n, d, ds, sa, counts, m, q, props))) => {
                (c.min(defined), rows, n, [d, ds, sa], m, q, props, counts)
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(mpsc::RecvError) => return Err("the engine thread ended during the load".into()),
        };
        // The seat counts its rounds and serves several slots: its `slots round` and `slot call`
        // records print from now on.
        if counts {
            crate::record::require(crate::record::groups::bind::SLOTS, "bind::spawn");
        }
        Ok(SeatEngine {
            link: Link {
                tx: Some(tx),
                rx,
                logits: bloomery_gpu::host_filled(n_vocab),
                scratch: Vec::new(),
            },
            worker: Some(worker),
            vocab,
            ctx_max,
            rows,
            slots,
            slot_drafts: drafts[0],
            drafts_sampled: drafts[1],
            snapshots_async: drafts[2],
            media,
            quantum,
            card,
            props,
            cache_ram,
            note: S::note,
            reply: None,
        })
    }

    /// The sequences the engine serves at once, the opened seat's answer
    /// ([`Seat::slots`]): one until the seat made resident slots.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }
}

/// How long the engine thread polls for the next command before it parks:
/// a decode step's reply and the server's next call cross in less, and a
/// parked thread's wake-up would sit on every token's path.
const RECEIVE_SPIN: Duration = Duration::from_millis(1);

/// The engine thread's next command: polled for until [`RECEIVE_SPIN`] has
/// passed since the call, then waited for parked. Commands sent before the
/// server's end are all received before the end is.
fn receive_command(rx: &Receiver<Cmd>) -> Result<Cmd, mpsc::RecvError> {
    let started = Instant::now();
    while started.elapsed() < RECEIVE_SPIN {
        match rx.try_recv() {
            Ok(cmd) => return Ok(cmd),
            Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvError),
            Err(mpsc::TryRecvError::Empty) => std::hint::spin_loop(),
        }
    }
    rx.recv()
}

impl Link {
    /// One command; a lent logits buffer comes back to the link.
    fn call(&mut self, cmd: Cmd) -> Result<u32, EngineError> {
        let reply = self.ask(cmd)?;
        if let Some(row) = reply.logits {
            self.logits = row;
        }
        reply.result.map_err(EngineError)
    }

    /// One command and its reply, the position left to the caller.
    fn ask(&self, cmd: Cmd) -> Result<Reply, EngineError> {
        self.send(cmd)?;
        self.reply()
    }

    /// One command sent; [`Link::reply`] waits for its answer.
    fn send(&self, cmd: Cmd) -> Result<(), EngineError> {
        match self.tx.as_ref().map(|tx| tx.send(cmd)) {
            Some(Ok(())) => Ok(()),
            _ => Err(EngineError("the engine thread is gone".to_owned())),
        }
    }

    /// The answer to the command sent last.
    fn reply(&self) -> Result<Reply, EngineError> {
        self.rx
            .recv()
            .map_err(|_| EngineError("the engine thread ended mid-call".to_owned()))
    }

    /// The selected slot's state as a value ([`Seat::snapshot`]), its
    /// failure's kind as [`snapshot_of`] read it.
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        let reply = self.ask(Cmd::Save)?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Saved(s)) => s,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a save with no state".to_owned(),
            ))),
        }
    }

    fn snapshot_size(&self) -> Result<Option<u64>, StateError> {
        let reply = self.ask(Cmd::SaveSize)?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::SaveSize(size)) => size,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a snapshot size with no size".to_owned(),
            ))),
        }
    }

    /// `state` back into the selected slot ([`Seat::resume`]), its failure's
    /// kind as [`resume_of`] read it.
    fn resume(&self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        let reply = self.ask(Cmd::Resume(Arc::clone(state)))?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Resumed(r)) => r,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a resume with no outcome".to_owned(),
            ))),
        }
    }

    /// The selected slot's state as a value whose copy may still run
    /// ([`Seat::snapshot_begin`]), its failure's kind as [`begin_of`] read it.
    fn snapshot_begin(&self) -> Result<Pending, StateError> {
        let reply = self.ask(Cmd::SaveBegin)?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Begun(p)) => p,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a begun save with no state".to_owned(),
            ))),
        }
    }

    /// A begun copy's answer from the engine thread: `wait` waits for it,
    /// else it is asked without waiting.
    fn flight(&self, p: &Pending, wait: bool) -> Result<Option<Landed>, StateError> {
        let cmd = if wait {
            Cmd::SaveWait(p.clone())
        } else {
            Cmd::SavePoll(p.clone())
        };
        let reply = self.ask(cmd)?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Polled(l)) => l,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a copy's flight with no outcome".to_owned(),
            ))),
        }
    }

    /// The first `n` positions of `state` back into the selected slot
    /// ([`Seat::resume_prefix`]), its failure's kind as [`resume_prefix_of`]
    /// read it.
    fn resume_prefix(&self, state: &Arc<dyn Saved>, n: usize) -> Result<usize, StateError> {
        let reply = self.ask(Cmd::ResumePrefix(Arc::clone(state), n))?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Prefix(r)) => r,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a prefix resume with no outcome".to_owned(),
            ))),
        }
    }

    /// The first `n` positions of slot `from` onto the selected slot
    /// ([`Seat::share_prefix`]), its failure's kind as [`share_of`] read it.
    fn share_prefix(&self, from: usize, n: usize) -> Result<SharedPrefix, StateError> {
        let reply = self.ask(Cmd::Share(from, n))?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Shared(r)) => r,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a share with no outcome".to_owned(),
            ))),
        }
    }

    /// The selected slot's state ([`Seat::save_state`]) into `out` through
    /// the pipe: the engine thread writes the chunks, this thread writes
    /// each to `out` as it comes. An error of `out` closes the pipe — the
    /// seat's next write fails — and is the answer ([`settle`]).
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        let (tx, chunks) = mpsc::sync_channel(PIPE_DEPTH);
        self.send(Cmd::SaveState(tx))?;
        let written = drain(chunks, out);
        let saved = state_of(self.reply()?, "save");
        settle(saved, written)
    }

    /// The selected slot's state replaced with the one `input` carries
    /// ([`Seat::restore_state`]), through the pipe: this thread reads `input`
    /// to its end a chunk at a time, the engine thread reads the chunks. The
    /// bytes the answer counts are the ones the seat read; the server holds
    /// them to the file's.
    fn restore_state(&self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        let (chunks, rx) = mpsc::sync_channel(PIPE_DEPTH);
        self.send(Cmd::RestoreState(rx))?;
        let read = pump(input, chunks);
        let restored = state_of(self.reply()?, "restore");
        settle(restored, read)
    }

    /// A step; with `logits_out`, the engine's buffer is lent for it and
    /// copied out once it is back.
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.call_lent(|logits| Cmd::Next { last, logits }, logits_out)
    }

    /// `ids` and the step of the last ([`Cmd::NextAfter`]); with
    /// `logits_out`, the engine's buffer is lent as a step's is.
    fn next_after(
        &mut self,
        ids: &[u32],
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let ids = ids.to_vec();
        self.call_lent(|logits| Cmd::NextAfter { ids, logits }, logits_out)
    }

    /// A prompt's last call and the step of its last id
    /// ([`Cmd::PrefillNext`]); with `logits_out`, the engine's buffer is lent
    /// as a step's is.
    fn prefill_next(
        &mut self,
        ids: &[u32],
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let ids = ids.to_vec();
        self.call_lent(|logits| Cmd::PrefillNext { ids, logits }, logits_out)
    }

    /// The command `cmd` makes of the engine's logits buffer, lent when the
    /// caller wants the row (`logits_out`) and copied out once it is back.
    fn call_lent(
        &mut self,
        cmd: impl FnOnce(Option<Vec<f32>>) -> Cmd,
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let lent = logits_out
            .is_some()
            .then(|| std::mem::take(&mut self.logits));
        let arg = self.call(cmd(lent))?;
        if let Some(out) = logits_out {
            if out.len() != self.logits.len() {
                return Err(EngineError(format!(
                    "the caller's logits buffer holds {}, the engine's {}",
                    out.len(),
                    self.logits.len()
                )));
            }
            out.copy_from_slice(&self.logits);
        }
        Ok(arg)
    }

    /// A pass into the caller's `out`, which crosses to the engine thread and
    /// back; it holds the kept tokens on success.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        let reply = self.ask(Cmd::Pass {
            last,
            out: std::mem::take(out),
        })?;
        match reply.extra {
            Extra::Pass(kept, d) => {
                *out = kept;
                reply.result.map(|_| d).map_err(EngineError)
            }
            _ => Err(EngineError(
                "the engine thread answered a pass with no tokens".to_owned(),
            )),
        }
    }

    /// A sampled pass into the caller's `out` ([`Cmd::PassSampled`]): the
    /// history and the sampler cross to the engine thread and come back with
    /// the reply, or with the command when the thread is gone before it took
    /// it. Only a thread that ended mid-call keeps the sampler: a stand-in
    /// that refuses by name comes back beside the error.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: serve::Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, serve::Sampler) {
        let gone = |why: &str| EngineError(why.to_owned());
        let cmd = Cmd::PassSampled {
            last,
            history,
            sampler,
            best,
            out: std::mem::take(out),
        };
        let sent = match self.tx.as_ref() {
            Some(tx) => tx.send(cmd).map_err(|mpsc::SendError(cmd)| cmd),
            None => Err(cmd),
        };
        if let Err(cmd) = sent {
            let Cmd::PassSampled {
                history,
                sampler,
                out: back,
                ..
            } = cmd
            else {
                unreachable!("the command that did not cross is the sampled pass built here")
            };
            *out = back;
            return (Err(gone("the engine thread is gone")), history, sampler);
        }
        let reply = match self.reply() {
            Ok(r) => r,
            Err(e) => return (Err(e), Vec::new(), lost_sampler()),
        };
        match reply.extra {
            Extra::PassSampled {
                out: taken,
                drafted,
                history,
                sampler,
            } => {
                *out = taken;
                (
                    reply.result.map(|_| drafted).map_err(EngineError),
                    history,
                    sampler,
                )
            }
            _ => (
                Err(gone(
                    "the engine thread answered a sampled pass with no ids",
                )),
                Vec::new(),
                lost_sampler(),
            ),
        }
    }

    /// A step of several slots in one command
    /// ([`SeatEngine`]'s `step_slots`): the engine thread selects and steps
    /// each row in order on the seat, and the answers and lent rows come
    /// back with the one reply — not two round trips a row. The rows' logits
    /// travel as the link's own scratch, one row a round, grown once and
    /// reused: no token allocates a row.
    fn step_slots(&mut self, rows: &mut [serve::SlotRow<'_>]) -> Result<(), EngineError> {
        let n_vocab = self.logits.len();
        // Two rows of one slot would step its sequence twice in the round,
        // the second on the first's answer: named before anything crosses.
        if let Some((i, r)) = rows
            .iter()
            .enumerate()
            .find(|(i, r)| rows[..*i].iter().any(|p| p.slot == r.slot))
        {
            return Err(EngineError(format!(
                "a round of {} rows names slot {} twice (row {i} again)",
                rows.len(),
                r.slot
            )));
        }
        // Grown, never shrunk: a shorter round keeps the rows it grew.
        if self.scratch.len() < rows.len() {
            self.scratch.resize_with(rows.len(), Vec::new);
        }
        let mut steps = Vec::with_capacity(rows.len());
        let mut refused = None;
        for (i, row) in rows.iter().enumerate() {
            // A row that wants the logits row borrows the link's scratch for
            // it; a caller buffer of another length is a named error before
            // anything crosses.
            let lent = row.logits.as_ref().map_or(Ok(None), |b| {
                if b.len() != n_vocab {
                    Err(EngineError(format!(
                        "a row's logits buffer holds {}, the vocabulary has {n_vocab}",
                        b.len()
                    )))
                } else {
                    let mut v = std::mem::take(&mut self.scratch[i]);
                    v.resize(n_vocab, 0.0);
                    Ok(Some(v))
                }
            });
            match lent {
                Ok(logits) => steps.push(SlotStep {
                    slot: row.slot,
                    last: row.last,
                    logits,
                    next: 0,
                }),
                Err(e) => {
                    refused = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = refused {
            self.take_back(&mut steps);
            return Err(e);
        }
        let reply = self.ask(Cmd::StepSlots(steps))?;
        let Extra::StepSlots(mut back) = reply.extra else {
            return Err(EngineError(
                "the engine thread answered a step of several slots with no rows".to_owned(),
            ));
        };
        for (i, (row, step)) in rows.iter_mut().zip(back.iter_mut()).enumerate() {
            row.next = step.next;
            if let (Some(v), Some(out)) = (step.logits.as_ref(), row.logits.as_deref_mut()) {
                out.copy_from_slice(v);
            }
            self.scratch[i] = step.logits.take().unwrap_or_default();
        }
        reply.result.map(|_| ()).map_err(EngineError)
    }

    /// The scratch rows `steps` hold back into the link, after a call that
    /// did not cross (a refused buffer length): what was taken out goes home.
    fn take_back(&mut self, steps: &mut [SlotStep]) {
        for (i, s) in steps.iter_mut().enumerate() {
            if s.logits.is_some() {
                self.scratch[i] = s.logits.take().unwrap_or_default();
            }
        }
    }

    /// One command for a round of several drafted passes
    /// ([`SeatEngine`]'s `advance_slots`): the engine thread selects and
    /// passes each row in order, on the seat, and each row's kept ids and
    /// drafted counts come back with the one reply — not a select and a pass
    /// round trip a row. The rows' kept-id buffers cross to the thread and
    /// back as the command's own rows.
    fn pass_slots(&mut self, rows: &mut [serve::SlotPass<'_>]) -> Result<(), EngineError> {
        // Two rows of one slot would pass its sequence twice in the round,
        // the second on the first's answer: named before anything crosses.
        if let Some((i, r)) = rows
            .iter()
            .enumerate()
            .find(|(i, r)| rows[..*i].iter().any(|p| p.slot == r.slot))
        {
            return Err(EngineError(format!(
                "a round of {} rows names slot {} twice (row {i} again)",
                rows.len(),
                r.slot
            )));
        }
        let mut sent = Vec::with_capacity(rows.len());
        for r in rows.iter_mut() {
            let draw = match r.draw.as_mut() {
                Some(d) => Some(RowDraw {
                    history: d.history.to_vec(),
                    sampler: d.sampler.take().ok_or_else(|| {
                        EngineError(format!(
                            "slot {}: a sampled row without its sampler",
                            r.slot
                        ))
                    })?,
                    best: d.best,
                }),
                None => None,
            };
            sent.push(SlotPassRow {
                slot: r.slot,
                last: r.last,
                out: std::mem::take(r.out),
                drafted: Drafted::default(),
                draw,
            });
        }
        let reply = self.ask(Cmd::PassSlots(sent))?;
        let Extra::PassSlots(mut back) = reply.extra else {
            return Err(EngineError(
                "the engine thread answered a pass of several slots with no rows".to_owned(),
            ));
        };
        let mut changed = None;
        for (row, back) in rows.iter_mut().zip(back.iter_mut()) {
            row.drafted = back.drafted;
            *row.out = std::mem::take(&mut back.out);
            if let (Some(to), Some(from)) = (row.draw.as_mut(), back.draw.take()) {
                if from.history != to.history {
                    changed = Some(row.slot);
                }
                *to.sampler = Some(from.sampler);
            }
        }
        reply.result.map_err(EngineError)?;
        if let Some(slot) = changed {
            return Err(EngineError(format!(
                "slot {slot}: a sampled pass handed back another history"
            )));
        }
        Ok(())
    }
}

/// Whether a feed of `n` ids may start at `g`'s position: at least one, and
/// all inside the context.
fn check_feed<S: Seat>(g: &S, n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("prefill: no ids to feed".into());
    }
    if g.pos() + n > g.ctx_max() {
        return Err(format!(
            "prefill: position {} + {n} ids exceed the context {}",
            g.pos(),
            g.ctx_max()
        ));
    }
    Ok(())
}

/// A pass on `g`, refused before it runs when its rows do not fit the
/// context; one that keeps no token or more than its rows is the seat's
/// defect, named.
fn pass<S: Seat + ?Sized>(g: &mut S, last: u32, out: &mut Vec<u32>) -> Result<Drafted, String> {
    let rows = g.pass_rows();
    if g.pos() + rows > g.ctx_max() {
        return Err(format!(
            "a pass of {rows} rows from position {} passes the context {}",
            g.pos(),
            g.ctx_max()
        ));
    }
    let d = g.pass(last, out).map_err(|e| e.to_string())?;
    if out.is_empty() || out.len() > rows {
        return Err(format!("a pass of {rows} rows kept {} tokens", out.len()));
    }
    Ok(d)
}

/// A sampled pass on `g` ([`Seat::pass_sampled`]), refused before it runs as
/// [`pass`] refuses one; a seat that keeps no id or more than its rows, or
/// hands `history` back changed, is the seat's defect, named.
fn pass_sampled<S: Seat + ?Sized>(
    g: &mut S,
    last: u32,
    history: &mut Vec<u32>,
    sampler: &mut serve::Sampler,
    best: Option<usize>,
    out: &mut Vec<u32>,
) -> Result<Drafted, String> {
    let rows = g.pass_rows();
    if g.pos() + rows > g.ctx_max() {
        return Err(format!(
            "a pass of {rows} rows from position {} passes the context {}",
            g.pos(),
            g.ctx_max()
        ));
    }
    let given = history.len();
    let d = g
        .pass_sampled(last, history, sampler, best, out)
        .map_err(|e| e.to_string())?;
    if out.is_empty() || out.len() > rows {
        return Err(format!("a pass of {rows} rows kept {} tokens", out.len()));
    }
    if history.len() != given {
        return Err(format!(
            "a sampled pass handed back a history of {} ids, given {given}",
            history.len()
        ));
    }
    Ok(d)
}

/// The sampler a [`Link::pass_sampled`] hands back when the engine thread
/// ended holding the request's own: it refuses by name if anything draws
/// from it.
fn lost_sampler() -> serve::Sampler {
    Box::new(|_: &[f32], _: &[u32]| -> u32 {
        panic!("this request's sampler ended with the engine thread")
    })
}

/// One command on the engine thread, and the logits buffer a `Next` was lent,
/// handed back whatever the result.
fn serve_cmd<S: Seat>(
    g: &mut S,
    cmd: Cmd,
    n_vocab: usize,
) -> (Result<u32, String>, Option<Vec<f32>>) {
    let at = g.pos();
    let result = match cmd {
        Cmd::Prefill(ids) => {
            let refuse =
                |e: String| format!("prefill of {} ids from position {at}: {e}", ids.len());
            check_feed(g, ids.len()).map_err(refuse).and_then(|()| {
                g.prefill_only(&ids)
                    .map(|()| 0)
                    .map_err(|e| refuse(e.to_string()))
            })
        }
        Cmd::PrefillMedia(ids, feeds) => {
            let refuse = |e: String| {
                format!(
                    "prefill of {} ids with {} image(s) from position {at}: {e}",
                    ids.len(),
                    feeds.len()
                )
            };
            check_feed(g, ids.len()).map_err(refuse).and_then(|()| {
                g.prefill_media(&ids, &feeds)
                    .map_err(|e| refuse(e.to_string()))
            })
        }
        Cmd::Next { last, mut logits } => {
            let arg = next_row(g, last, logits.as_deref_mut(), n_vocab);
            return (arg, logits);
        }
        Cmd::NextAfter { ids, mut logits } => {
            let arg = prefill_row(g, &ids, logits.as_deref_mut(), n_vocab, S::prefill_row);
            return (arg, logits);
        }
        Cmd::PrefillNext { ids, mut logits } => {
            let arg = prefill_row(g, &ids, logits.as_deref_mut(), n_vocab, S::prefill_next);
            return (arg, logits);
        }
        Cmd::Reset => g
            .reset()
            .map(|()| 0)
            .map_err(|e| format!("reset at position {at}: {e}")),
        Cmd::Select(slot) => g
            .select(slot)
            .map(|()| 0)
            .map_err(|e| format!("select of slot {slot}: {e}")),
        Cmd::Rollback(pos) if pos as usize == at => Ok(0),
        Cmd::Rollback(pos) => g
            .rollback(pos)
            .map(|()| 0)
            .map_err(|e| format!("rollback to position {pos} from {at}: {e}")),
        Cmd::Keep(..)
        | Cmd::Splits { .. }
        | Cmd::Save
        | Cmd::SaveSize
        | Cmd::SaveBegin
        | Cmd::SavePoll(_)
        | Cmd::SaveWait(_)
        | Cmd::Resume(_)
        | Cmd::ResumePrefix(..)
        | Cmd::ShareKeep(..)
        | Cmd::Share(..)
        | Cmd::SaveState(_)
        | Cmd::RestoreState(_)
        | Cmd::Pass { .. }
        | Cmd::PassSampled { .. }
        | Cmd::StepSlots(..)
        | Cmd::PassSlots(..)
        | Cmd::ResidencyReset
        | Cmd::Close => Err(
            "a keep, split, pass, step or pass of several slots, save, resume, slot file's save \
             or restore, residency reset or close of the seat reached the step loop; the \
             engine thread answers it"
                .to_owned(),
        ),
        Cmd::Pos => Ok(0),
    };
    (result, None)
}

/// One step and, into `row`, the target's logits of it ([`Seat::step_row`]).
/// The row is checked here: one of the wrong length is refused before the
/// step runs, and one holding a NaN is the step's error, not the sampler's to
/// absorb.
fn next_row<S: Seat>(
    g: &mut S,
    last: u32,
    row: Option<&mut [f32]>,
    n_vocab: usize,
) -> Result<u32, String> {
    if let Some(row) = row.as_ref()
        && row.len() != n_vocab
    {
        let at = g.pos();
        return Err(format!(
            "a logits buffer of {} at position {at}, the vocabulary has {n_vocab}",
            row.len()
        ));
    }
    next_lent_row(g, last, row)
}

/// [`Seat::prefill_row`]'s default: [`Seat::prefill_only`] of all but the last of
/// `ids` (none for one id) and [`Seat::step`], or [`Seat::step_row`] into
/// `row`, of the last.
pub fn prefill_then_step<S: Seat + ?Sized>(
    g: &mut S,
    ids: &[u32],
    row: Option<&mut [f32]>,
) -> Result<u32, GateError> {
    let Some((&last, run)) = ids.split_last() else {
        return Err("a prefill and its step of no ids".into());
    };
    if !run.is_empty() {
        g.prefill_only(run)?;
    }
    match row {
        Some(row) => g.step_row(last, row),
        None => g.step(last),
    }
}

/// `ids` and the step of the last by `call` ([`Seat::prefill_row`] or
/// [`Seat::prefill_next`]), checked as a prefill's feed and a step's row are:
/// a row of the wrong length or a feed past the context is refused before the
/// call runs, and a row holding a NaN is the call's error.
fn prefill_row<S: Seat>(
    g: &mut S,
    ids: &[u32],
    row: Option<&mut [f32]>,
    n_vocab: usize,
    call: impl FnOnce(&mut S, &[u32], Option<&mut [f32]>) -> Result<u32, GateError>,
) -> Result<u32, String> {
    let at = g.pos();
    let refuse = |e: String| format!("{} ids and a step from position {at}: {e}", ids.len());
    if let Some(row) = row.as_ref()
        && row.len() != n_vocab
    {
        return Err(refuse(format!(
            "a logits buffer of {}, the vocabulary has {n_vocab}",
            row.len()
        )));
    }
    check_feed(g, ids.len()).map_err(refuse)?;
    let Some(row) = row else {
        return call(g, ids, None).map_err(|e| refuse(e.to_string()));
    };
    let arg = call(g, ids, Some(&mut *row)).map_err(|e| refuse(e.to_string()))?;
    if let Some(i) = row.iter().position(|v| v.is_nan()) {
        return Err(refuse(format!("logit {i} is NaN")));
    }
    Ok(arg)
}

/// [`next_row`] over a row the link already checked against the vocabulary
/// and lent as its own scratch, so the length guard there cannot fire: the
/// ctx-full and NaN guards stay, as a round's rows need them too
/// ([`step_rows_in_turn`]).
fn next_lent_row<S: Seat + ?Sized>(
    g: &mut S,
    last: u32,
    row: Option<&mut [f32]>,
) -> Result<u32, String> {
    let at = g.pos();
    if at >= g.ctx_max() {
        return Err(format!(
            "step at position {at}: step: the context {} is full",
            g.ctx_max()
        ));
    }
    let Some(row) = row else {
        return g
            .step(last)
            .map_err(|e| format!("step at position {at}: {e}"));
    };
    let arg = g
        .step_row(last, row)
        .map_err(|e| format!("step at position {at}: {e}"))?;
    if let Some(i) = row.iter().position(|v| v.is_nan()) {
        return Err(format!("logit {i} is NaN after position {at}"));
    }
    Ok(arg)
}

/// The fallback round of several slots' steps ([`Seat::step_slots`]'s
/// default): a select and a step a row, in row order — one command, not two
/// round trips a row. A row that fails ends the round there; the error is
/// the server's to die on. A seat that counts its rounds
/// ([`Seat::step_stats`]) prints the `slots round` record: the fallback's
/// passes are its rows.
pub fn step_rows_in_turn<S: Seat + ?Sized>(g: &mut S, rows: &mut [SlotStep]) -> Result<(), String> {
    let mut failed = None;
    for r in rows.iter_mut() {
        let run = g
            .select(r.slot)
            .map_err(|e| e.to_string())
            .and_then(|()| next_lent_row(g, r.last, r.logits.as_deref_mut()));
        match run {
            Ok(next) => r.next = next,
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    if failed.is_none() && g.step_stats() {
        crate::record::slots_round("step", rows.len(), rows.len(), g.slots()).eprint();
    }
    failed.map_or(Ok(()), Err)
}

/// The fallback round of several slots' drafted passes ([`Seat::pass_slots`]'s
/// default): a select and a pass a row, in row order. A row that fails ends
/// the round there; the error is the server's to die on. A seat that counts
/// its rounds ([`Seat::step_stats`]) prints the `slots round` record: the
/// fallback's passes are its rows.
pub fn pass_rows_in_turn<S: Seat + ?Sized>(
    g: &mut S,
    rows: &mut [SlotPassRow],
) -> Result<(), String> {
    let mut failed = None;
    for r in rows.iter_mut() {
        let at = g.pos();
        let run = g.select(r.slot).map_err(|e| e.to_string()).and_then(|()| {
            r.out.clear();
            match r.draw.as_mut() {
                None => pass(g, r.last, &mut r.out),
                Some(d) => pass_sampled(
                    g,
                    r.last,
                    &mut d.history,
                    &mut d.sampler,
                    d.best,
                    &mut r.out,
                ),
            }
            .map_err(|e| format!("pass at position {at}: {e}"))
        });
        match run {
            Ok(d) => r.drafted = d,
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    if failed.is_none() && g.step_stats() {
        crate::record::slots_round("pass", rows.len(), rows.len(), g.slots()).eprint();
    }
    failed.map_or(Ok(()), Err)
}

/// The server's call of one slot `cmd` is, as its `slot call` record names
/// it ([`crate::record::slot_call`]): `step` its `next`, `pass` its
/// `advance`, `sampled` its `advance_sampled`; `None` for every other
/// command.
fn one_slot_call(cmd: &Cmd) -> Option<&'static str> {
    match cmd {
        Cmd::Next { .. } | Cmd::PrefillNext { .. } => Some("step"),
        Cmd::Pass { .. } => Some("pass"),
        Cmd::PassSampled { .. } => Some("sampled"),
        _ => None,
    }
}

/// A seat that counts its rounds ([`Seat::step_stats`]) and serves several
/// slots prints the `slot call` record of each call of one slot it ran,
/// `call` ([`one_slot_call`]), on `selected`, the slot the server's last
/// select named: beside the `slots round` record of a round of several, the
/// log shows how each of the server's rows ran. A call with no slot named
/// since a round of several breaks the server's select contract
/// (`serve::Engine::step_slots`) and is refused by name: the record would
/// guess its slot.
fn count_call<S: Seat + ?Sized>(g: &S, call: &str, selected: Option<usize>) -> Result<(), String> {
    if !g.step_stats() || g.slots() < 2 {
        return Ok(());
    }
    let slot = selected.ok_or_else(|| {
        format!(
            "a {call} of one slot after a round of several slots with no select since: the \
             server selects before its next call of one slot"
        )
    })?;
    crate::record::slot_call(call, slot, g.slots()).eprint();
    Ok(())
}

/// The selected slot's state as a value ([`Seat::snapshot`]), on the engine
/// thread; a failure with its kind ([`seat_failure`]).
fn snapshot_of<S: Seat + ?Sized>(g: &mut S) -> Result<Arc<dyn Saved>, StateError> {
    let at = g.pos();
    g.snapshot()
        .map_err(|e| seat_failure(&*e, format!("snapshot at position {at}")))
}

fn snapshot_size_of<S: Seat + ?Sized>(g: &mut S) -> Result<Option<u64>, StateError> {
    let at = g.pos();
    g.snapshot_size()
        .map_err(|e| seat_failure(&*e, format!("snapshot size at position {at}")))
}

/// `state` back into the selected slot ([`Seat::resume`]), on the engine
/// thread; a failure with its kind ([`seat_failure`]).
fn resume_of<S: Seat + ?Sized>(g: &mut S, state: &dyn Saved) -> Result<(), StateError> {
    g.resume(state).map_err(|e| {
        seat_failure(
            &*e,
            format!("resume of a state of {} positions", state.n_tokens()),
        )
    })
}

/// The selected slot's state as a value whose copy may still run
/// ([`Seat::snapshot_begin`]), on the engine thread; a failure with its kind
/// ([`seat_failure`]).
fn begin_of<S: Seat + ?Sized>(g: &mut S) -> Result<Pending, StateError> {
    let at = g.pos();
    g.snapshot_begin()
        .map_err(|e| seat_failure(&*e, format!("snapshot begun at position {at}")))
}

/// The first `n` positions of `state` back into the selected slot
/// ([`Seat::resume_prefix`]), on the engine thread; a failure with its kind
/// ([`seat_failure`]).
fn resume_prefix_of<S: Seat + ?Sized>(
    g: &mut S,
    state: &dyn Saved,
    n: usize,
) -> Result<usize, StateError> {
    g.resume_prefix(state, n).map_err(|e| {
        seat_failure(
            &*e,
            format!(
                "resume of the first {n} positions of a state of {} positions",
                state.n_tokens()
            ),
        )
    })
}

/// The first `n` positions of slot `from` onto the selected slot
/// ([`Seat::share_prefix`]), on the engine thread; a failure with its kind
/// ([`seat_failure`]).
fn share_of<S: Seat + ?Sized>(
    g: &mut S,
    from: usize,
    n: usize,
) -> Result<SharedPrefix, StateError> {
    g.share_prefix(from, n).map_err(|e| {
        seat_failure(
            &*e,
            format!("share of the first {n} positions of slot {from}"),
        )
    })
}

/// A seat's failure of a prompt-cache state with `what` before it, as the
/// server's kind: a staging pool with no room for the state
/// (`bloomery_gpu::checkpoint::staging::NoRoom` anywhere in the chain) is
/// `NoRoom`, its three numbers one to one, which the server answers by giving
/// up its landed states, oldest first, until the copy fits; a refusal by name
/// (a `GpuError` of kind `State` or `Shape` anywhere in the chain,
/// [`Seat::snapshot`]) is `Format`, the request's, which the server answers
/// by keeping no state or by resetting the slot; any other failure — the
/// card's, the driver's, a fault, an error of no known kind — is `Engine`,
/// which ends it.
pub fn seat_failure(e: &(dyn std::error::Error + 'static), what: String) -> StateError {
    let chain = || std::iter::successors(Some(e), |e| e.source());
    if let Some(short) = chain().find_map(|e| e.downcast_ref::<NoRoom>()) {
        return StateError::NoRoom {
            need: short.need,
            held: short.held,
            lent: short.lent,
        };
    }
    let refused = chain()
        .find_map(|e| e.downcast_ref::<bloomery_gpu::GpuError>())
        .is_some_and(|g| {
            matches!(
                g,
                bloomery_gpu::GpuError::State { .. } | bloomery_gpu::GpuError::Shape { .. }
            )
        });
    if refused {
        StateError::Format(format!("{what}: {e}"))
    } else {
        StateError::Engine(EngineError(format!("{what}: {e}")))
    }
}

/// A copy on a seat's save lane, as the cache's entry for its state waits for
/// it ([`Cached`]) and the server's [`Flight`] polls it.
pub trait Landing: Send + Sync {
    /// Whether the copy has landed; never waits.
    fn ready(&self) -> Result<bool, GpuError>;
    /// Waits for the copy to land and returns what it cost; the same answer
    /// on every call.
    fn land(&self) -> Result<Landed, GpuError>;
}

impl<T: Send + Sync> Landing for Saving<T> {
    fn ready(&self) -> Result<bool, GpuError> {
        Saving::ready(self)
    }

    fn land(&self) -> Result<Landed, GpuError> {
        let c = Saving::land(self)?;
        Ok(Landed {
            copy_ms: c.copy_ms,
            calls: c.calls,
            alloc_ms: c.alloc_ms,
        })
    }
}

/// The server's flight of a copy on a seat's lane: its failure's kind as
/// [`seat_failure`] reads it.
struct Flying(Arc<dyn Landing>);

impl Flying {
    fn failed(e: &GpuError) -> StateError {
        seat_failure(e, "the copy of a saved state".to_owned())
    }
}

impl Flight for Flying {
    fn poll(&self) -> Result<Option<Landed>, StateError> {
        if self.0.ready().map_err(|e| Flying::failed(&e))? {
            self.0.land().map(Some).map_err(|e| Flying::failed(&e))
        } else {
            Ok(None)
        }
    }

    fn wait(&self) -> Result<Landed, StateError> {
        self.0.land().map_err(|e| Flying::failed(&e))
    }
}

/// A body's sequence state `S` as the server's prompt cache holds it, with the
/// seat's own side `D` of it (the draft's parked side, or `()`): the one
/// [`Saved`] of every seat whose state sits on `bloomery_gpu`'s `SeqView`.
/// The cache ranks it by the body's rule, every position it holds or the
/// point it carries at or below the shared prefix. A state taken on a save
/// lane ([`begin`]) may still be copying: the body's side is reached only
/// through [`Cached::body`], which waits for the copy first, so no resume
/// reads a state the lane is writing.
pub struct Cached<S, D> {
    body: S,
    /// The seat's side of the state, taken on the engine thread at the save.
    pub draft: D,
    landing: Option<Arc<dyn Landing>>,
}

impl<S: SeqView, D: Send + Sync + 'static> Cached<S, D> {
    /// A state taken on the request path: landed already.
    pub fn landed(body: S, draft: D) -> Cached<S, D> {
        Cached {
            body,
            draft,
            landing: None,
        }
    }

    /// `state` as this seat's, refused by name for a state another seat took
    /// (`whose` names the seat's body).
    pub fn of<'a>(state: &'a dyn Saved, whose: &str) -> Result<&'a Cached<S, D>, GateError> {
        state
            .as_any()
            .downcast_ref::<Cached<S, D>>()
            .ok_or_else(|| format!("a saved state that is not {whose}").into())
    }

    /// The body's state once its copy has landed, waited for here: the one
    /// way to read it.
    pub fn body(&self) -> Result<&S, GateError> {
        if let Some(l) = &self.landing {
            l.land()?;
        }
        Ok(&self.body)
    }
}

impl<S: SeqView, D: Send + Sync + 'static> Saved for Cached<S, D> {
    fn n_tokens(&self) -> usize {
        self.body.positions() as usize
    }

    fn n_bytes(&self) -> u64 {
        self.body.bytes() as u64
    }

    fn keepable(&self, n: usize) -> usize {
        self.body.keep_point(u32::try_from(n).unwrap_or(u32::MAX)) as usize
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// [`Seat::snapshot_begin`] over a body's save on its lane: `copy` is what the
/// body's async save answered, its state and the copy's [`Landing`] or the
/// pool's [`NoRoom`], which goes up as the seat's error for the server to
/// read ([`seat_failure`]: evict, retry, then the request path); `side` is
/// the seat's part of the state, taken once the copy is begun.
pub fn begin<S: SeqView, D: Send + Sync + 'static>(
    copy: Result<(S, Arc<dyn Landing>), NoRoom>,
    side: impl FnOnce() -> Result<D, GateError>,
) -> Result<Pending, GateError> {
    let (body, landing) = copy.map_err(GateError::from)?;
    let cached = Cached {
        body,
        draft: side()?,
        landing: Some(Arc::clone(&landing)),
    };
    Ok(Pending::new(Arc::new(cached), Arc::new(Flying(landing))))
}

/// What a seat gives [`put_back`]: the body's state and the seat's side of
/// it as [`Cached`] holds them, and the three steps of the seat's draft that
/// go around the body's restore.
pub trait PutBack {
    /// The body's sequence state, as [`Cached`] holds it.
    type Body: SeqView;
    /// The seat's side of a saved state (the draft's parked side, or `()`).
    type Side: Send + Sync + 'static;

    /// Refused by name unless a state whose side is `side` fits this seat: a
    /// state saved with the draft on put back with it off, or the other way.
    fn takes(&self, side: &Self::Side) -> Result<(), GateError>;
    /// The session reset, and the selected slot's draft started over.
    fn reset_selected(&mut self) -> Result<(), GateError>;
    /// `side` back in the selected slot's draft: it joins the slot's next
    /// call where it left, or stays off for its parked why.
    fn unpark(&mut self, side: &Self::Side) -> Result<(), GateError>;
}

/// `state`, which a seat's [`Cached`] took (`whose` names its body), put back
/// into the selected slot: the state's copy waited for ([`Cached::body`]), the
/// seat's side checked ([`PutBack::takes`]), the session reset
/// ([`PutBack::reset_selected`]), `put` run on the seat with the body's state
/// (the model's whole or prefix restore, the positions it leaves the model
/// at), then the seat's side put back ([`PutBack::unpark`]). Refused by name
/// before the reset for a state another seat took or a side this seat does not
/// take; the body's own refusals come from `put`. The positions `put` left.
pub fn put_back<P: PutBack>(
    seat: &mut P,
    state: &dyn Saved,
    whose: &str,
    put: impl FnOnce(&mut P, &P::Body) -> Result<u32, GpuError>,
) -> Result<usize, GateError> {
    let saved = Cached::<P::Body, P::Side>::of(state, whose)?;
    let body = saved.body()?;
    seat.takes(&saved.draft)?;
    seat.reset_selected()?;
    let at = put(seat, body)?;
    seat.unpark(&saved.draft)?;
    Ok(at as usize)
}

/// The selected slot's state ([`Seat::save_state`]) written into the pipe
/// `chunks`, on the engine thread: its last chunk sent and its end dropped
/// before the reply, so the server's thread holds every byte when the
/// answer reaches it. The positions the seat stands at and the bytes it
/// wrote.
fn save_into<S: Seat + ?Sized>(
    g: &mut S,
    chunks: SyncSender<Vec<u8>>,
) -> Result<SavedState, StateError> {
    let at = g.pos();
    let mut w = PipeWriter {
        chunks,
        chunk: Vec::new(),
    };
    let n_bytes = g
        .save_state(&mut w)
        .map_err(|e| at_position(e, "save", at))?;
    w.flush()?;
    Ok(SavedState {
        n_tokens: at,
        n_bytes,
    })
}

/// The selected slot's state replaced from the pipe `chunks`
/// ([`Seat::restore_state`]), on the engine thread, the pipe's end dropped
/// before the reply: the positions restored and the bytes the seat read.
fn restore_from<S: Seat + ?Sized>(
    g: &mut S,
    chunks: Receiver<Vec<u8>>,
) -> Result<SavedState, StateError> {
    let at = g.pos();
    let mut r = PipeReader {
        chunks,
        chunk: Vec::new(),
        at: 0,
        bytes: 0,
    };
    let n_tokens = g
        .restore_state(&mut r)
        .map_err(|e| at_position(e, "restore", at))?;
    Ok(SavedState {
        n_tokens,
        n_bytes: r.bytes,
    })
}

/// `e` with the command and the position the seat stood at when it is the
/// engine's; a refusal or a stream's error is the request's answer as the
/// seat gave it.
fn at_position(e: StateError, what: &str, at: usize) -> StateError {
    match e {
        StateError::Engine(EngineError(m)) => {
            StateError::Engine(EngineError(format!("{what} at position {at}: {m}")))
        }
        other => other,
    }
}

/// A `SaveState` or `RestoreState` reply's outcome, its kind as the seat
/// gave it; a reply without one is the engine thread's defect, named.
fn state_of(reply: Reply, what: &str) -> Result<SavedState, StateError> {
    match (reply.result, reply.extra) {
        (Ok(_), Extra::State(r)) => r,
        (Err(e), _) => Err(StateError::Engine(EngineError(e))),
        (Ok(_), _) => Err(StateError::Engine(EngineError(format!(
            "the engine thread answered a slot file's {what} with no outcome"
        )))),
    }
}

/// The answer of a slot file's state that crossed the pipe: the seat's
/// engine failure first (a failed model is never hidden behind the file),
/// then this thread's error of the stream (which closed the pipe, and so
/// caused whatever the seat answered after it), then the seat's answer.
fn settle(
    seat: Result<SavedState, StateError>,
    stream: io::Result<()>,
) -> Result<SavedState, StateError> {
    match (seat, stream) {
        (Err(e @ StateError::Engine(_)), _) => Err(e),
        (_, Err(e)) => Err(StateError::Io(e)),
        (r, Ok(())) => r,
    }
}

/// Every chunk of a save's pipe into `out`, until the engine thread drops
/// its end; the first error of `out` drops this end (the seat's next write
/// fails) and is returned.
fn drain(chunks: Receiver<Vec<u8>>, out: &mut dyn Write) -> io::Result<()> {
    for chunk in &chunks {
        out.write_all(&chunk)?;
    }
    Ok(())
}

/// `input` to its end into a restore's pipe, [`PIPE_CHUNK`] bytes a chunk,
/// this end dropped after it: the end of `input` reaches the seat as the
/// pipe's. A seat that stops reading drops its end, and the pump stops
/// there. An error of `input` is returned.
fn pump(input: &mut dyn Read, chunks: SyncSender<Vec<u8>>) -> io::Result<()> {
    loop {
        let mut chunk = Vec::with_capacity(PIPE_CHUNK);
        if (&mut *input)
            .take(PIPE_CHUNK as u64)
            .read_to_end(&mut chunk)?
            == 0
        {
            return Ok(());
        }
        if chunks.send(chunk).is_err() {
            return Ok(());
        }
    }
}

/// The engine thread's side of a save's pipe: bytes gathered into a chunk
/// of [`PIPE_CHUNK`], each full chunk sent, the last one by `flush`. A pipe
/// the server's thread closed is `BrokenPipe`.
struct PipeWriter {
    chunks: SyncSender<Vec<u8>>,
    chunk: Vec<u8>,
}

impl PipeWriter {
    fn send(&mut self) -> io::Result<()> {
        let chunk = std::mem::take(&mut self.chunk);
        self.chunks.send(chunk).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the server stopped taking the slot's state",
            )
        })
    }
}

impl Write for PipeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.chunk.capacity() == 0 {
            self.chunk.reserve_exact(PIPE_CHUNK);
        }
        let n = buf.len().min(PIPE_CHUNK - self.chunk.len());
        self.chunk.extend_from_slice(&buf[..n]);
        if self.chunk.len() == PIPE_CHUNK {
            self.send()?;
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.chunk.is_empty() {
            Ok(())
        } else {
            self.send()
        }
    }
}

/// The engine thread's side of a restore's pipe: the chunks the server's
/// thread read, in order, and the pipe's end as the stream's. `bytes`
/// counts what the seat read.
struct PipeReader {
    chunks: Receiver<Vec<u8>>,
    chunk: Vec<u8>,
    at: usize,
    bytes: u64,
}

impl Read for PipeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        while self.at == self.chunk.len() {
            match self.chunks.recv() {
                Ok(c) => {
                    self.chunk = c;
                    self.at = 0;
                }
                Err(mpsc::RecvError) => return Ok(0),
            }
        }
        let n = buf.len().min(self.chunk.len() - self.at);
        buf[..n].copy_from_slice(&self.chunk[self.at..self.at + n]);
        self.at += n;
        self.bytes += n as u64;
        Ok(n)
    }
}

impl Engine for SeatEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.vocab.clone()
    }

    /// The seat's ([`Seat::media_model`]), read once at the load: a shared
    /// handle, the same one every request thread prepares images on.
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.media.clone()
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.link.call(Cmd::Prefill(ids.to_vec())).map(|_| ())
    }

    /// The seat's ([`Seat::prefill_media`]) as one command on the engine
    /// thread; an empty feed takes [`Engine::prefill`]'s plain command.
    fn prefill_media(&mut self, ids: &[u32], media: &[MediaFeed]) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Ok(());
        }
        if media.is_empty() {
            return self.prefill(ids);
        }
        self.link
            .call(Cmd::PrefillMedia(ids.to_vec(), media.to_vec()))
            .map(|_| ())
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.link.next(last, logits_out)
    }

    /// One command on the engine thread ([`Seat::prefill_next`]) for a call
    /// that carries no image; one that does is [`Engine::prefill_media`] of
    /// all but the last id and [`Engine::next`] of the last.
    fn prefill_next(
        &mut self,
        ids: &[u32],
        media: &[MediaFeed],
        marks: &[usize],
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let Some((&last, run)) = ids.split_last() else {
            return Err(EngineError("prefill_next of no ids".to_owned()));
        };
        if media.is_empty() {
            return self.link.prefill_next(ids, logits_out);
        }
        self.prefill_marked(run, media, marks)?;
        self.next(last, logits_out)
    }

    /// The seat's ([`Seat::prompt_quantum`]), read once at the load.
    fn prompt_quantum(&self) -> Option<NonZeroUsize> {
        self.quantum
    }

    /// One command on the engine thread ([`Seat::prefill_row`]).
    fn next_after(
        &mut self,
        ids: &[u32],
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        if ids.is_empty() {
            return Err(EngineError("next_after of no ids".to_owned()));
        }
        self.link.next_after(ids, logits_out)
    }

    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.link.pass(last, out)
    }

    fn advance_rows(&self) -> usize {
        self.rows
    }

    /// One command ([`Cmd::PassSampled`]): the seat's sampled pass on the
    /// engine thread, the history and sampler back with the reply.
    fn advance_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: serve::Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, serve::Sampler) {
        self.link.pass_sampled(last, history, sampler, best, out)
    }

    /// The seat's declaration ([`Seat::drafts_sampled`]), read once at the
    /// spawn.
    fn drafts_sampled(&self) -> bool {
        self.drafts_sampled
    }

    /// The seat's declaration ([`Seat::snapshots_async`]), read once at the
    /// spawn.
    fn snapshots_async(&self) -> bool {
        self.snapshots_async
    }

    /// The seat's slots ([`Seat::slots`]), read once at the spawn: the
    /// server serves this many sequences at once and steps them together.
    fn slots(&self) -> usize {
        self.slots
    }

    /// The seat's slot ([`Seat::select`]), forwarded to the thread: the
    /// exchange happens there, with every other call the seat runs.
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.link.call(Cmd::Select(slot)).map(|_| ())
    }

    /// One command for a round of several rows ([`serve::Engine`'s
    /// `step_slots`]): the thread selects and steps each row in order, on
    /// the seat, and the answers and requested logits rows come back with
    /// the one reply — not a select and a step round trip a row.
    fn step_slots(&mut self, rows: &mut [serve::SlotRow<'_>]) -> Result<(), EngineError> {
        self.link.step_slots(rows)
    }

    /// One command for a round of several drafted passes
    /// ([`serve::Engine`]'s `advance_slots`): the thread selects and passes
    /// each row in order, on the seat, and each row's kept ids and drafted
    /// counts come back with the one reply — not a select and a pass round
    /// trip a row.
    fn advance_slots(&mut self, rows: &mut [serve::SlotPass<'_>]) -> Result<(), EngineError> {
        self.link.pass_slots(rows)
    }

    /// The seat's declaration ([`Seat::slot_drafts`]), read once at the
    /// spawn: the server serves several slots of a drafting engine only when
    /// it holds.
    fn slot_drafts(&self) -> bool {
        self.slot_drafts
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.link.call(Cmd::Reset).map(|_| ())
    }

    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, EngineError> {
        let reply = self.link.ask(Cmd::ResidencyReset)?;
        reply.result.map_err(EngineError)?;
        match reply.extra {
            Extra::Residency(r) => Ok(r),
            _ => Err(EngineError(
                "the engine thread answered a residency reset with no report".to_owned(),
            )),
        }
    }

    fn close(&mut self) -> Result<(), EngineError> {
        self.link.call(Cmd::Close).map(|_| ())
    }

    /// The body's rule for the request's reply, asked on the engine thread
    /// (`spawn`'s `keep_for`). A
    /// thread that does not answer grants nothing: the caller resets, and the
    /// next call reports the thread.
    fn will_reply(&mut self, tokens: Option<usize>) {
        self.reply = tokens;
    }

    fn keepable(&self, n: usize) -> usize {
        match self.link.ask(Cmd::Keep(n, self.reply)).map(|r| r.result) {
            Ok(Ok(k)) => (k as usize).min(n),
            _ => 0,
        }
    }

    /// The body's rule, as `keepable` asks it; a thread that does not answer
    /// is named as the reason.
    fn keep_limit(&self, n: usize) -> Option<String> {
        match self.link.ask(Cmd::Keep(n, self.reply)) {
            Ok(Reply {
                extra: Extra::Why(why),
                ..
            }) => why,
            Ok(_) => Some("the engine thread answered a keep query with no rule".to_owned()),
            Err(e) => Some(e.0),
        }
    }

    fn cache_ram(&self) -> u64 {
        self.cache_ram
    }

    fn snapshot_size(&self) -> Result<Option<u64>, StateError> {
        self.link.snapshot_size()
    }

    /// The seat's state ([`Link::snapshot`]): a refusal by name is the
    /// request's (the server keeps no state), any other failure fatal.
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        self.link.snapshot()
    }

    /// The state back into the seat ([`Link::resume`]): a refusal by name
    /// is the request's (the server drops the state and resets the slot),
    /// any other failure fatal.
    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        self.link.resume(state)
    }

    /// The seat's state with its copy in flight ([`Link::snapshot_begin`]).
    fn snapshot_begin(&self) -> Result<Pending, StateError> {
        self.link.snapshot_begin()
    }

    /// Whether `pending`'s copy has landed, asked on the engine thread.
    fn snapshot_poll(&self, pending: &Pending) -> Result<Option<Landed>, StateError> {
        self.link.flight(pending, false)
    }

    /// The wait for `pending`'s copy, on the engine thread.
    fn snapshot_wait(&self, pending: &Pending) -> Result<Landed, StateError> {
        self.link.flight(pending, true)?.ok_or_else(|| {
            StateError::Engine(EngineError(
                "the engine thread answered a wait for a copy with none landed".to_owned(),
            ))
        })
    }

    /// The first `n` positions of the state back into the seat
    /// ([`Link::resume_prefix`]).
    fn resume_prefix(&mut self, state: &Arc<dyn Saved>, n: usize) -> Result<usize, StateError> {
        self.link.resume_prefix(state, n)
    }

    /// The seat's share rule ([`Seat::share_keep`]), asked on the engine
    /// thread; a thread that does not answer grants nothing.
    fn share_keepable(&self, from: usize, n: usize) -> usize {
        match self.link.ask(Cmd::ShareKeep(from, n)).map(|r| r.result) {
            Ok(Ok(k)) => (k as usize).min(n),
            _ => 0,
        }
    }

    /// Slot `from`'s prefix onto the selected slot ([`Link::share_prefix`]).
    fn share_prefix(&mut self, from: usize, n: usize) -> Result<SharedPrefix, StateError> {
        self.link.share_prefix(from, n)
    }

    /// The selected slot's state through the pipe ([`Seat::save_state`]),
    /// its kind carried: `Unsupported` is the server's 501, a refusal or
    /// the file's error the request's, an engine failure fatal.
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        self.link.save_state(out)
    }

    /// [`Engine::save_state`]'s way back ([`Seat::restore_state`]); a
    /// refusal leaves the slot to the server, which resets it.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        self.link.restore_state(input)
    }

    /// The body's cuts; a thread that does not answer cuts nowhere, and the
    /// next call reports the thread.
    fn prefill_splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let cmd = Cmd::Splits {
            first,
            end,
            marks: marks.to_vec(),
        };
        match self.link.ask(cmd) {
            Ok(Reply {
                extra: Extra::Splits(at),
                ..
            }) => at,
            _ => Vec::new(),
        }
    }

    fn note(&self, note: &CacheNote) {
        (self.note)(note);
    }

    /// A cut to where the model stands is nothing to do: the thread answers
    /// it without a rollback.
    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        let pos =
            u32::try_from(n).map_err(|_| EngineError(format!("cut to position {n}: past u32")))?;
        self.link.call(Cmd::Rollback(pos)).map(|_| ())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        match self.link.ask(Cmd::Pos) {
            Ok(r) => format!("card={} position={}", self.card, r.pos),
            Err(e) => format!("card={} position unknown ({})", self.card, e.0),
        }
    }

    fn props_engine(&self) -> EngineProps {
        self.props.clone()
    }

    /// The process's engine waits ([`bloomery_gpu::waits`]), read here with
    /// no call to the seat's thread: the seat is the process's one engine.
    fn waits(&self) -> Waits {
        Waits {
            card: bloomery_gpu::waits::card_waited(),
            ple: bloomery_gpu::waits::ple_waited(),
        }
    }
}

impl Drop for SeatEngine {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop; the model is dropped there.
        self.link.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cmd, Link, Reply};
    use std::sync::mpsc;

    const N_VOCAB: usize = 8;

    // PIN(2026-10-21): Removed seat-source/copy wiring assertions; the shared binder reads parsed levers.

    /// A link to a thread that answers every `Next` as the engine thread does:
    /// it fills the lent row and hands it back with the argmax. A
    /// `StepSlots` it answers the same way, a row at a time in row order, and
    /// a `PassSlots` keeps `last` and one drafted id a row, a sampled row's
    /// its sampler's draw given its history.
    fn echo_link(alter_history: Option<usize>) -> (Link, std::thread::JoinHandle<()>) {
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (replies, rx) = mpsc::channel::<Reply>();
        let worker = std::thread::spawn(move || {
            let received = std::iter::from_fn(|| super::receive_command(&cmds).ok());
            for (at, cmd) in received.enumerate() {
                let reply = match cmd {
                    Cmd::Next { last, mut logits } => {
                        if let Some(row) = logits.as_deref_mut() {
                            row.fill(last as f32);
                        }
                        Reply {
                            result: Ok(last + 1),
                            logits,
                            pos: at + 1,
                            extra: super::Extra::None,
                        }
                    }
                    Cmd::StepSlots(mut rows) => {
                        for r in rows.iter_mut() {
                            if let Some(row) = r.logits.as_deref_mut() {
                                row.fill(r.last as f32);
                            }
                            r.next = r.last + 1;
                        }
                        Reply {
                            result: Ok(0),
                            logits: None,
                            pos: at + 1,
                            extra: super::Extra::StepSlots(rows),
                        }
                    }
                    Cmd::PassSlots(mut rows) => {
                        for r in rows.iter_mut() {
                            let next = match r.draw.as_mut() {
                                Some(d) => (d.sampler)(&[], &d.history),
                                None => r.last + 1,
                            };
                            if alter_history == Some(r.slot) {
                                let d = r.draw.as_mut().expect("the altered row samples");
                                d.history[0] += 1;
                            }
                            r.out = vec![r.last, next];
                            r.drafted = serve::Drafted {
                                proposed: 1,
                                accepted: 1,
                            };
                        }
                        Reply {
                            result: Ok(0),
                            logits: None,
                            pos: at + 1,
                            extra: super::Extra::PassSlots(rows),
                        }
                    }
                    _ => panic!("the echo thread answers Next, StepSlots and PassSlots only"),
                };
                if replies.send(reply).is_err() {
                    return;
                }
            }
        });
        let link = Link {
            tx: Some(tx),
            rx,
            logits: vec![0.0; N_VOCAB],
            scratch: Vec::new(),
        };
        (link, worker)
    }

    /// The engine thread's receiver ([`super::receive_command`]) takes a
    /// command that arrives long after its poll has ended, parked meanwhile
    /// (its thread's CPU time over a wait far longer than the poll stays
    /// near the poll's), and every command sent before the sender's end
    /// before it reports the end.
    #[test]
    fn the_command_receiver_parks_and_takes_every_command_before_the_end() {
        let thread_cpu = || {
            let stat = std::fs::read_to_string("/proc/thread-self/schedstat")
                .expect("the thread's schedstat");
            let ns = stat.split_whitespace().next().expect("its run time");
            std::time::Duration::from_nanos(ns.parse().expect("a count of ns"))
        };
        let idle = std::time::Duration::from_millis(200);
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (done, ended) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let before = thread_cpu();
            let mut taken = 0usize;
            while let Ok(cmd) = super::receive_command(&cmds) {
                assert!(matches!(cmd, Cmd::Reset), "the commands as sent");
                taken += 1;
            }
            done.send((taken, thread_cpu() - before))
                .expect("the test waits");
        });
        tx.send(Cmd::Reset).expect("the receiver runs");
        std::thread::sleep(idle);
        tx.send(Cmd::Reset).expect("the receiver runs");
        tx.send(Cmd::Reset).expect("the receiver runs");
        drop(tx);
        let (taken, cpu) = ended
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the receiver reports the end");
        worker.join().expect("the receiver thread");
        assert_eq!(taken, 3, "every command before the end");
        assert!(
            cpu < idle / 4,
            "the receiver parked through the wait: {cpu:?} of CPU over {idle:?}"
        );
    }

    /// A round of several slots is one command whose reply sets each row's
    /// answer in row order ([`Link::step_slots`]), the rows' logits reading
    /// into the link's own scratch — grown once on the first round, reused
    /// every round, so no token allocates a row; a caller buffer of another
    /// length and a slot named twice are named errors.
    #[test]
    fn a_round_of_slots_answers_in_row_order() {
        use serve::SlotRow;
        let (mut link, worker) = echo_link(None);
        let mut a = [0.0f32; N_VOCAB];
        let mut b = [0.0f32; N_VOCAB];
        let mut rows = vec![
            SlotRow {
                slot: 0,
                last: 3,
                logits: Some(&mut a),
                next: 0,
            },
            SlotRow {
                slot: 1,
                last: 5,
                logits: None,
                next: 0,
            },
            SlotRow {
                slot: 2,
                last: 9,
                logits: Some(&mut b),
                next: 0,
            },
        ];
        link.step_slots(&mut rows).expect("a round of three slots");
        assert_eq!(
            rows.iter().map(|r| r.next).collect::<Vec<_>>(),
            vec![4, 6, 10],
            "each row its own answer, in row order"
        );
        assert_eq!(a, [3.0f32; N_VOCAB]);
        assert_eq!(b, [9.0f32; N_VOCAB]);
        assert_eq!(link.scratch.len(), 3, "the scratch came home");
        let first = link.scratch[0].as_ptr();
        let third = link.scratch[2].as_ptr();
        let mut c = [0.0f32; N_VOCAB];
        let mut rows = vec![
            SlotRow {
                slot: 1,
                last: 7,
                logits: Some(&mut c),
                next: 0,
            },
            SlotRow {
                slot: 0,
                last: 1,
                logits: None,
                next: 0,
            },
        ];
        link.step_slots(&mut rows).expect("a second, shorter round");
        assert_eq!(rows[0].next, 8);
        assert_eq!(rows[1].next, 2);
        assert_eq!(c, [7.0f32; N_VOCAB]);
        assert_eq!(link.scratch[0].as_ptr(), first, "the scratch reused");
        assert_eq!(
            link.scratch[2].as_ptr(),
            third,
            "an untouched row kept its row"
        );
        // A caller buffer of another length is a named error, and the
        // scratch it had lent comes home for the next round.
        let mut short = [0.0f32; N_VOCAB - 1];
        let mut rows = vec![
            SlotRow {
                slot: 0,
                last: 2,
                logits: Some(&mut short),
                next: 0,
            },
            SlotRow {
                slot: 1,
                last: 4,
                logits: None,
                next: 0,
            },
        ];
        let e = link
            .step_slots(&mut rows)
            .expect_err("a short buffer among the rows");
        assert!(e.0.contains("holds 7"), "{e}");
        // Two rows of one slot are a named error before anything crosses.
        let mut rows = vec![
            SlotRow {
                slot: 1,
                last: 2,
                logits: None,
                next: 0,
            },
            SlotRow {
                slot: 1,
                last: 4,
                logits: None,
                next: 0,
            },
        ];
        let e = link
            .step_slots(&mut rows)
            .expect_err("one slot twice in a round");
        assert!(e.0.contains("names slot 1 twice"), "{e}");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A round of several drafted passes is one command whose reply gives
    /// each row its own kept ids and drafted counts in row order
    /// ([`Link::pass_slots`]), the caller's buffers crossing with the
    /// command; a sampled row's history crosses with it and its sampler comes
    /// back with its state moved; a slot named twice is a named error.
    #[test]
    fn a_round_of_passes_answers_in_row_order() {
        use serve::SlotPass;
        let (mut link, worker) = echo_link(None);
        let mut a = Vec::new();
        let mut b = Vec::new();
        let history = [7u32, 7, 7];
        let mut draws = 0;
        let mut sampler: Option<serve::Sampler> = Some(Box::new(move |_: &[f32], h: &[u32]| {
            draws += 1;
            100 * draws + h.len() as u32
        }));
        let mut rows = vec![
            SlotPass {
                slot: 1,
                last: 3,
                out: &mut a,
                drafted: serve::Drafted::default(),
                draw: None,
            },
            SlotPass {
                slot: 0,
                last: 5,
                out: &mut b,
                drafted: serve::Drafted::default(),
                draw: Some(serve::SlotDraw {
                    history: &history,
                    sampler: &mut sampler,
                    best: None,
                }),
            },
        ];
        link.pass_slots(&mut rows).expect("a round of two passes");
        let counts = rows.iter().map(|r| r.drafted).collect::<Vec<_>>();
        drop(rows);
        assert_eq!(a, vec![3, 4], "the first row's kept ids");
        assert_eq!(b, vec![5, 103], "the sampled row's draw, given its history");
        let mut back = sampler.expect("the sampled row's sampler came back");
        assert_eq!(back(&[], &[]), 200, "the sampler's state moved one draw");
        assert_eq!(
            counts,
            vec![
                serve::Drafted {
                    proposed: 1,
                    accepted: 1
                };
                2
            ],
            "each row its own counts, in row order"
        );
        // Two rows of one slot are a named error before anything crosses.
        let mut c = Vec::new();
        let mut d = Vec::new();
        let mut rows = vec![
            SlotPass {
                slot: 2,
                last: 2,
                out: &mut c,
                drafted: serve::Drafted::default(),
                draw: None,
            },
            SlotPass {
                slot: 2,
                last: 4,
                out: &mut d,
                drafted: serve::Drafted::default(),
                draw: None,
            },
        ];
        let e = link
            .pass_slots(&mut rows)
            .expect_err("one slot twice in a round of passes");
        assert!(e.0.contains("names slot 2 twice"), "{e}");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A same-length history edit in a successful round reply is refused
    /// after every row's sampler and output have come home.
    #[test]
    fn a_round_rejects_a_changed_history_and_restores_every_sampler() {
        let (mut link, worker) = echo_link(Some(1));
        let histories = [[7, 7, 7], [9, 9, 9]];
        let mut samplers: [Option<serve::Sampler>; 2] = std::array::from_fn(|_| {
            let mut draws = 0u32;
            Some(Box::new(move |_: &[f32], _: &[u32]| {
                draws += 1;
                draws
            }) as serve::Sampler)
        });
        let mut outputs = [Vec::new(), Vec::new()];
        let mut rows: Vec<_> = outputs
            .iter_mut()
            .zip(&mut samplers)
            .zip(&histories)
            .enumerate()
            .map(|(i, ((out, sampler), history))| serve::SlotPass {
                slot: 1 - i,
                last: 3,
                out,
                drafted: serve::Drafted::default(),
                draw: Some(serve::SlotDraw {
                    history,
                    sampler,
                    best: None,
                }),
            })
            .collect();
        let result = link.pass_slots(&mut rows);
        drop(rows);
        link.tx = None;
        worker.join().expect("the echo thread");
        let e = result.expect_err("a same-length history edit must fail");
        assert_eq!(e.0, "slot 1: a sampled pass handed back another history");
        assert_eq!(outputs, [vec![3, 1], vec![3, 1]]);
        for sampler in samplers {
            assert_eq!(sampler.expect("every sampler comes back")(&[], &[]), 2);
        }
    }

    /// Every sampled step reads into the buffer the link was built with: the
    /// row crosses to the engine thread and back, and no token allocates one.
    #[test]
    fn sampled_steps_reuse_one_logits_buffer() {
        let (mut link, worker) = echo_link(None);
        let built = link.logits.as_ptr();
        let mut out = [0.0f32; N_VOCAB];
        for last in [3u32, 5, 9] {
            let arg = link.next(last, Some(&mut out)).expect("a step");
            assert_eq!(arg, last + 1);
            assert_eq!(out, [last as f32; N_VOCAB], "the row of step {last}");
            assert_eq!(
                link.logits.as_ptr(),
                built,
                "step {last} read into another allocation than the link's"
            );
            assert_eq!(link.logits.len(), N_VOCAB);
        }
        let greedy = link.next(11, None).expect("a greedy step");
        assert_eq!(greedy, 12);
        assert_eq!(link.logits.as_ptr(), built, "a greedy step kept the buffer");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A caller's buffer of another length is a named error, not a partial copy.
    #[test]
    fn a_short_caller_buffer_is_refused() {
        let (mut link, worker) = echo_link(None);
        let mut short = [0.0f32; N_VOCAB - 1];
        let e = link.next(3, Some(&mut short)).expect_err("a short buffer");
        assert!(e.0.contains("holds 7, the engine's 8"), "{e}");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A MiB value past u64 bytes names the flag it parsed
    /// ([`CacheRam::parse_mib`]): the park flag's, not the cache's.
    #[test]
    fn a_mib_value_past_u64_names_the_flag_it_parsed() {
        let e = super::CacheRam::parse_mib("--park-ram", &(1u64 << 44).to_string())
            .expect_err("2^44 MiB passes u64 bytes");
        let e = e.to_string();
        assert!(e.contains("--park-ram 17592186044416 MiB"), "{e}");
        assert!(!e.contains("--cache-ram"), "{e}");
    }

    /// The `parallel` line names the shape a round of several slots runs
    /// in: one slot is `one` whatever the body does, several slots are `one`
    /// where the body runs one pass of the busy rows and `turns` where it
    /// does not, and `together` is the several-slots one-pass fact the seats
    /// size their rounds by.
    #[test]
    fn the_parallel_line_names_the_shape_of_a_round() {
        use super::{Pass, SlotsFrom, parallel_text};
        for (slots, one_pass, want) in [
            (1, true, Pass::One),
            (1, false, Pass::One),
            (2, true, Pass::One),
            (2, false, Pass::Turns),
        ] {
            assert_eq!(Pass::of(slots, one_pass), want, "slots {slots} {one_pass}");
        }
        assert!(!Pass::One.together(1));
        assert!(Pass::One.together(2));
        assert!(!Pass::Turns.together(2));
        assert_eq!(Pass::One.word(), "one");
        assert_eq!(Pass::Turns.word(), "turns");
        // Every `from=` word the seats print, as the gates read it.
        for (from, word) in [
            (SlotsFrom::Flag, "flag"),
            (SlotsFrom::Ctx, "ctx"),
            (SlotsFrom::Default, "default"),
            (SlotsFrom::RouteTrace, "route-trace"),
            (SlotsFrom::Dspark, "dspark"),
        ] {
            assert_eq!(
                parallel_text(2, 7, from, Pass::Turns),
                format!("parallel rule=slots slots=2 slot_ctx=7 total=14 from={word} pass=turns")
            );
        }
        assert_eq!(
            parallel_text(2, 4096, SlotsFrom::Flag, Pass::One),
            "parallel rule=slots slots=2 slot_ctx=4096 total=8192 from=flag pass=one"
        );
        // The line is no record kind; the start screen (`serve::screen`)
        // reads it by key, and its fixed starts carry it as printed.
        for (start, line) in [
            (
                include_str!("../../serve/tests/fixtures/screen-start.log"),
                parallel_text(2, 16384, SlotsFrom::Default, Pass::One),
            ),
            (
                include_str!("../../serve/tests/fixtures/screen-start-qwen38.log"),
                parallel_text(1, 8192, SlotsFrom::Ctx, Pass::One),
            ),
        ] {
            assert!(
                start.lines().any(|l| l == line),
                "the screen's fixed start lacks `{line}`"
            );
        }
    }

    /// The slot count and its word: `--parallel` as given, one slot at the
    /// whole context beside a set `--ctx` (the flag is one request's
    /// context, not a total the seat's slots split), the seat's own default
    /// with neither, and a count of none refused by name.
    #[test]
    fn slots_of_rows() {
        use super::{NoSlot, SlotsFrom, slots_of};
        type Row = (Option<usize>, bool, usize, (usize, SlotsFrom));
        let rows: [Row; 6] = [
            (Some(3), false, 2, (3, SlotsFrom::Flag)),
            (Some(3), true, 2, (3, SlotsFrom::Flag)),
            (Some(1), true, 2, (1, SlotsFrom::Flag)),
            (None, true, 2, (1, SlotsFrom::Ctx)),
            (None, false, 2, (2, SlotsFrom::Default)),
            (None, false, 1, (1, SlotsFrom::Default)),
        ];
        for (parallel, ctx_set, seat, want) in rows {
            let got = slots_of(parallel, ctx_set, seat).expect("a count");
            assert_eq!(got, want, "parallel {parallel:?}, ctx set {ctx_set}");
        }
        for (parallel, ctx_set, seat) in [(Some(0), false, 2), (Some(0), true, 2), (None, false, 0)]
        {
            let refused = slots_of(parallel, ctx_set, seat).expect_err("no slot");
            assert_eq!(refused, NoSlot);
            assert_eq!(
                refused.to_string(),
                "--parallel 0: the server serves no slot"
            );
        }
    }

    /// The slot flags beside `--ctx`: `--slot-ctx S` asks `S x N` over `--parallel N` (one
    /// slot with no count), `--parallel auto` keeps its slot context for the seat, and each
    /// conflict or zero is refused by name; a seat that serves a count refuses auto.
    /// Mutants: `--slot-ctx` read as the total, auto beside `--ctx-size` taken as a total.
    #[test]
    fn slot_flags_ask_rows() {
        use super::{AutoSlots, SlotAsk, SlotFlags};
        let _note = crate::start::slot_note_tests();
        let ask = |flags: &[(&str, &str)], ctx: Option<usize>| {
            let mut slots = SlotFlags::default();
            for (f, v) in flags {
                assert!(slots.take(f, v).expect("a value"), "{f} is a slot flag");
            }
            slots.ask(ctx).map_err(|e| e.to_string())
        };
        let given = |ctx, parallel| Ok(SlotAsk::Given { ctx, parallel });
        let auto = |slot_ctx| Ok(SlotAsk::Auto(AutoSlots { slot_ctx }));
        assert_eq!(ask(&[], None), given(None, None));
        assert_eq!(
            ask(&[("-np", "4")], Some(32_768)),
            given(Some(32_768), Some(4))
        );
        assert_eq!(
            ask(&[("--parallel", "4"), ("--slot-ctx", "8192")], None),
            given(Some(32_768), Some(4))
        );
        assert_eq!(
            ask(&[("--slot-ctx", "8192")], None),
            given(Some(8192), None)
        );
        assert_eq!(ask(&[("--parallel", "auto")], None), auto(None));
        assert_eq!(
            ask(&[("--slot-ctx", "8192"), ("-np", "auto")], None),
            auto(Some(8192))
        );
        for (flags, ctx, named) in [
            (&[("--parallel", "0")][..], None, "--parallel 0"),
            (&[("--slot-ctx", "0")][..], None, "--slot-ctx 0"),
            (
                &[("--slot-ctx", "4096")][..],
                Some(4096),
                "--slot-ctx 4096 beside --ctx-size",
            ),
            (
                &[("--parallel", "auto")][..],
                Some(4096),
                "--parallel auto beside --ctx-size",
            ),
            (
                &[("--parallel", "2"), ("--slot-ctx", &usize::MAX.to_string())][..],
                None,
                "passes usize",
            ),
        ] {
            let e = ask(flags, ctx).expect_err("refused");
            assert!(e.contains(named), "{flags:?}: {e}");
        }
        let mut slots = SlotFlags::default();
        assert!(!slots.take("--ctx-size", "8").expect("not a slot flag"));
        assert!(slots.take("--parallel", "two").is_err());
        slots.take("--parallel", "auto").expect("auto");
        let e = slots.given(None).expect_err("auto refused").to_string();
        assert!(
            e.contains("--parallel auto: this seat serves the count"),
            "{e}"
        );
    }

    /// The context flag the parser reads, settings-file values included, is the one
    /// [`SlotFlags::ask`] notes and the one-request line names; replaying the retry through the
    /// parser asks the offered context.
    #[test]
    fn effective_context_flag_names_the_line_and_replays_the_retry() {
        use super::{SlotAsk, SlotFlags, one_request_line};
        let parse = |args: &[String]| {
            let mut flags = SlotFlags::default();
            let mut ctx = None;
            for (flag, value) in crate::llama_launch::serve_args(args) {
                let value = value.expect("a flag value");
                if flag == "--ctx-size" {
                    ctx = Some(value.parse::<usize>().expect("context"));
                } else {
                    assert!(flags.take(flag, value).expect("a valid flag"), "{flag}");
                }
            }
            let asked = flags.ask(ctx).expect("compatible slot flags");
            (crate::start::slot_ctx_noted(), asked)
        };
        let _note = crate::start::slot_note_tests();
        let settings = |args: &[String], json: &str| {
            crate::serve_config::joined(args, |_| Ok(json.to_owned()))
                .expect("settings")
                .0
        };
        for (raw, json, flag, offered, asked) in [
            (
                &["--config", "retry.json"][..],
                r#"{"parallel":"auto","slot-ctx":16384}"#,
                "--slot-ctx",
                8192,
                SlotAsk::Auto(super::AutoSlots {
                    slot_ctx: Some(8192),
                }),
            ),
            (
                &["--parallel", "auto", "--slot-ctx", "16384"][..],
                "{}",
                "--slot-ctx",
                8192,
                SlotAsk::Auto(super::AutoSlots {
                    slot_ctx: Some(8192),
                }),
            ),
            (
                &["--parallel", "4", "--ctx-size", "65536"][..],
                "{}",
                "--ctx-size",
                32_768,
                SlotAsk::Given {
                    ctx: Some(32_768),
                    parallel: Some(4),
                },
            ),
            (
                &["--parallel", "4", "--slot-ctx", "16384"][..],
                "{}",
                "--slot-ctx",
                8192,
                SlotAsk::Given {
                    ctx: Some(32_768),
                    parallel: Some(4),
                },
            ),
        ] {
            let raw: Vec<String> = raw.iter().map(|&a| a.to_owned()).collect();
            let (per_slot, _) = parse(&settings(&raw, json));
            assert_eq!(per_slot, flag == "--slot-ctx", "{raw:?}");
            assert!(
                one_request_line(offered, per_slot).starts_with(flag),
                "{raw:?}"
            );
            let (_, mut retried) = crate::llama_launch::take_serve_args(&raw, &[flag]);
            retried.extend([flag.to_owned(), offered.to_string()]);
            assert_eq!(parse(&settings(&retried, json)).1, asked, "{raw:?}");
        }
    }

    /// The `from=` words are read as text by the gates: each variant prints
    /// its word, `Display` and [`SlotsFrom::word`] alike, and no two are one.
    #[test]
    fn slots_from_words() {
        use super::SlotsFrom;
        let words = [
            (SlotsFrom::Flag, "flag"),
            (SlotsFrom::Ctx, "ctx"),
            (SlotsFrom::Default, "default"),
            (SlotsFrom::RouteTrace, "route-trace"),
            (SlotsFrom::Dspark, "dspark"),
            (SlotsFrom::Auto, "auto"),
        ];
        for (from, word) in words {
            assert_eq!(from.word(), word);
            assert_eq!(from.to_string(), word);
        }
        for (i, (_, a)) in words.iter().enumerate() {
            for (_, b) in &words[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    /// The checkpoints a load's sequences pin grow one budget a sequence
    /// ([`checkpoints_pinned`]), the bytes the machine's checkpoint reserve
    /// carries: a point of 1 MiB spaced 512 apart takes 9 points over a slot
    /// of 4,096 positions (a whole call's 8 and the previous call's end), and
    /// sequences that take none pin none.
    #[test]
    fn checkpoints_pinned_grow_a_budget_a_sequence() {
        use crate::residency38::{Seqs, checkpoints_pinned};
        use runtime::seqstate::Price;
        let price = Price {
            point_bytes: 1 << 20,
            every: 512,
        };
        let budget = 9 << 20;
        let pinned = |slots| checkpoints_pinned(Seqs::priced(slots, price), 4_096);
        assert_eq!(pinned(0).expect("a reserve"), 0);
        assert_eq!(pinned(1).expect("a reserve"), budget);
        assert_eq!(pinned(3).expect("a reserve"), 3 * budget);
        assert_eq!(pinned(usize::MAX).expect("a reserve"), u64::MAX);
        assert_eq!(
            checkpoints_pinned(Seqs::none(3), 4_096).expect("a reserve"),
            0
        );
        let past = u64::from(u32::MAX) + 1;
        let e = checkpoints_pinned(Seqs::priced(1, price), past).expect_err("a context past u32");
        assert!(e.to_string().contains("passes the u32 positions"), "{e}");
    }

    /// The unset residency rule's room is what the reading leaves past the
    /// plan's host need ([`HostNeed::room_left`]), the need holding the checkpoints
    /// the load's machine reserves ([`reserve_checkpoints`]) once: a churn
    /// pool that fits past the need of a load that declares none does not
    /// fit past the same plan's need on a machine that reserves two slots',
    /// and the pick is `off`, naming the room the reserve leaves: an unset
    /// word climbs on every body, and this pool fits at no P.
    #[test]
    fn the_residency38_room_reads_the_reserved_need() {
        use crate::residency38::{Seqs, checkpoints_reserved, reserve_checkpoints};
        use model::placement::workstation::{HostNeed, HostRead};
        use model::placement::{Host, Machine, Node};
        use runtime::seqstate::Price;
        // Points of 1 GiB spaced 512 apart: a slot of 4,096 positions takes
        // eight and the host budget fits four, 4 GiB a slot.
        let price = Price {
            point_bytes: 1 << 30,
            every: 512,
        };
        let slot = price.budget(4_096).expect("a budget");
        assert_eq!(slot, 4 << 30);
        // A plan of one card layer of 100 experts, its host the rest: the
        // unset rule pins none (P 0), the pool a fixed 10 GiB whatever P is, on
        // the Qwen3.8 body (`Body38::UNHOSTED_POOL`).
        let pick = |mem_left: i128| {
            bloomery_levers::residency38_at_plan(
                bloomery_levers::UnhostedPool::Carried,
                [100u64],
                384,
                |_pinned: usize| Ok::<u64, std::convert::Infallible>(10u64 << 30),
                i128::MAX,
                mem_left,
            )
            .expect("the pick")
        };
        let declared = |seqs: Seqs| {
            let mut machine = Machine {
                cards: Vec::new(),
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: 0,
                        reserves: Vec::new(),
                    },
                    unified: None,
                }],
            };
            reserve_checkpoints(&mut machine, seqs, 4_096).expect("a reserve");
            checkpoints_reserved(&machine)
        };
        // The plan's own terms, 12 GiB, and the reserve the need adds; the
        // reading is a given room, which the climb holds as given.
        let (available, terms) = (24u64 << 30, 12u64 << 30);
        let mem_left = |bytes: u64| {
            HostNeed {
                experts: bytes,
                tables: 0,
                shadows: 0,
                reserves: 0,
                extra: 0,
                read_ahead: 0,
                arena: 0,
                cards: 0,
                os: 0,
                card_budget: false,
            }
            .room_left(&(available, HostRead::Given))
        };
        let none = terms + declared(Seqs::none(2));
        assert_eq!(pick(mem_left(none)).pinned, Some(0));
        let two = terms + declared(Seqs::priced(2, price));
        assert_eq!(two - none, 2 * slot);
        // PIN(2026-10-13): the pool past the reserved need turns the word
        // `off` (`PoolUnhostable`): the pool is the same at every P here, so
        // the climb finds no count that fits. An unset word that kept P 0
        // with the pool unhosted decoded far slower than the climb.
        let short = pick(mem_left(two));
        assert_eq!(short.pinned, None);
        assert_eq!(short.pool(), bloomery_levers::ChurnHosting::Hosted);
        let bloomery_levers::Residency38Why::PoolUnhostable(pool) = short.why else {
            panic!(
                "the pool past the reserved need fits no count: {:?}",
                short.why
            );
        };
        assert_eq!(pool.needs, 10u64 << 30);
        assert_eq!(
            pool.mem_left,
            i128::from(available) - i128::from(terms) - i128::from(2 * slot)
        );
        assert_eq!(pool.mem_left, i128::from(available) - i128::from(two));
    }

    /// The nemotron, raon1 and tiered seats read `--place` through
    /// [`SeatFlags`], which reads it as [`place_word`] does: every word
    /// `Place::parse` takes parses to the placement it names, a list of tier
    /// cards stays the seat's plan to refuse (`None`), a body that serves a
    /// count refuses more by name, and a word no card or ordinal spells is
    /// refused.
    #[test]
    fn the_seats_read_the_place_word_as_one_grammar() {
        use super::{SeatFlags, place_word};
        use model::placement::workstation::{A6000, RTX_3090};
        let _note = crate::start::slot_note_tests();
        let usage = "usage";
        let seat = |word: &str| {
            let args = ["--place".to_owned(), word.to_owned()];
            SeatFlags::parse(&args, usage, |_, _| Ok(false))
        };
        for (word, name, cards) in [
            ("a", "a", vec![A6000.name]),
            ("gate", "gate", vec![RTX_3090.name]),
            ("bp", "bp", vec![A6000.name, RTX_3090.name]),
            ("a6000+3090", "bp", vec![A6000.name, RTX_3090.name]),
            ("3090+a6000", "3090+a6000", vec![RTX_3090.name, A6000.name]),
            ("cuda1+0", "cuda1+0", vec!["cuda1", "cuda0"]),
            ("tp", "tp", vec![A6000.name]),
        ] {
            let p = place_word(word, None).unwrap_or_else(|e| panic!("{word}: {e}"));
            assert_eq!((p.name(), p.cards()), (name, cards), "{word}");
            let flags = seat(word).unwrap_or_else(|e| panic!("{word}: {e}"));
            assert_eq!(flags.place, Some(p), "{word}");
        }
        let many = "cuda0+cuda1+cuda2";
        assert_eq!(place_word(many, None).expect(many).tier_cards().len(), 2);
        assert_eq!(
            seat(many).expect(many).place.map(|p| p.tier_cards().len()),
            Some(2)
        );
        place_word("bp", Some(("body", 1))).expect("one tier card");
        let e = place_word(many, Some(("body", 1)))
            .expect_err(many)
            .to_string();
        assert_eq!(
            e,
            "--place cuda0+cuda1+cuda2: 2 expert tier cards, and the body body serves 1 at most"
        );
        for refused in [place_word("b2", None).map(|_| ()), seat("b2").map(|_| ())] {
            let e = refused.expect_err("b2").to_string();
            assert!(e.starts_with("--place is a, gate, bp"), "{e}");
        }
    }

    /// The seats that read their flags through [`SeatFlags`] take `--slot-save-path` as one flag:
    /// unset it is `None`, which refuses every slot action, and given it is the directory; with no
    /// value the flag is refused by name.
    #[test]
    fn the_seats_take_the_slot_save_path_through_the_common_flags() {
        use super::SeatFlags;
        let _note = crate::start::slot_note_tests();
        let parse = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|a| (*a).to_owned()).collect();
            SeatFlags::parse(&args, "usage", |_, _| Ok(false))
        };
        assert_eq!(parse(&[]).expect("unset").slot_save_path, None);
        assert_eq!(
            parse(&["--slot-save-path", "/slots"])
                .expect("given")
                .slot_save_path,
            Some(std::path::PathBuf::from("/slots"))
        );
        let Err(e) = parse(&["--slot-save-path"]) else {
            panic!("a --slot-save-path with no value was taken");
        };
        assert!(
            e.to_string().starts_with("--slot-save-path needs a value"),
            "{e}"
        );
    }

    mod media {
        use std::sync::Arc;

        use serve::media::{ImageKey, MediaFeed};

        use super::super::{Cmd, Seat, serve_cmd};
        use crate::GateError;

        /// One recorded call: the ids it took, each feed's position and key.
        type Took = Vec<(Vec<u32>, Vec<(usize, u8)>)>;

        /// A feed of a two-position span whose image is keyed `key`, at `at`.
        fn feed(at: usize, key: u8) -> MediaFeed {
            MediaFeed {
                at,
                key: ImageKey([key; 32]),
                prepared: Arc::new(vision::Prepared::of_grid(
                    2,
                    vision::Patches {
                        plan: vision::GridPlan {
                            n_llm_h: 1,
                            n_llm_w: 1,
                            best_h: 14,
                            best_w: 14,
                        },
                        n_vit_h: 1,
                        n_vit_w: 1,
                        patch_len: 3,
                        bf16: vec![0; 3],
                    },
                )),
            }
        }

        /// A seat that takes images: its `prefill_media` records the call it
        /// took (the ids and each feed's position and key) and answers 7; its
        /// `prefill` answers 5; each call advances `pos` past its ids, as a
        /// seat does. Without the override, [`NoImages`] holds the trait's
        /// defaults over the same two.
        struct Images {
            pos: usize,
            took: Took,
        }

        impl Seat for Images {
            fn pos(&self) -> usize {
                self.pos
            }
            fn ctx_max(&self) -> usize {
                1 << 10
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
                self.took.push((ids.to_vec(), Vec::new()));
                self.pos += ids.len();
                Ok(5)
            }
            fn prefill_media(
                &mut self,
                ids: &[u32],
                feeds: &[MediaFeed],
            ) -> Result<u32, GateError> {
                self.took.push((
                    ids.to_vec(),
                    feeds.iter().map(|f| (f.at, f.key.0[0])).collect::<Vec<_>>(),
                ));
                self.pos += ids.len();
                Ok(7)
            }
            fn step(&mut self, _: u32) -> Result<u32, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn logits_into(&self, _: &mut [f32]) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn reset(&mut self) -> Result<(), GateError> {
                Ok(())
            }
            fn rollback(&mut self, _: u32) -> Result<(), GateError> {
                Ok(())
            }
            fn keep(&self, _: usize) -> (usize, Option<String>) {
                (0, None)
            }
            fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
                Vec::new()
            }
            fn snapshot(&mut self) -> Result<Arc<dyn serve::Saved>, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn resume(&mut self, _: &dyn serve::Saved) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn note(_: &serve::CacheNote) {}
        }

        /// [`Images`] with the trait's `prefill_media` default: the same
        /// record shape tells which path each call took.
        struct NoImages {
            took: Took,
        }

        impl NoImages {
            /// The seat over a fresh record.
            fn new() -> NoImages {
                NoImages { took: Vec::new() }
            }
        }

        impl Seat for NoImages {
            fn pos(&self) -> usize {
                0
            }
            fn ctx_max(&self) -> usize {
                1 << 10
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
                self.took.push((ids.to_vec(), Vec::new()));
                Ok(5)
            }
            fn step(&mut self, _: u32) -> Result<u32, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn logits_into(&self, _: &mut [f32]) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn reset(&mut self) -> Result<(), GateError> {
                Ok(())
            }
            fn rollback(&mut self, _: u32) -> Result<(), GateError> {
                Ok(())
            }
            fn keep(&self, _: usize) -> (usize, Option<String>) {
                (0, None)
            }
            fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
                Vec::new()
            }
            fn snapshot(&mut self) -> Result<Arc<dyn serve::Saved>, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn resume(&mut self, _: &dyn serve::Saved) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn note(_: &serve::CacheNote) {}
        }

        /// A `PrefillMedia` command is the seat's own `prefill_media` with
        /// its ids and each feed's position and key, and the seat's answer is
        /// the reply's; a call past the context is refused before it runs,
        /// naming the call and the position.
        #[test]
        fn a_prefill_media_command_reaches_the_seat() {
            let mut g = Images {
                pos: 3,
                took: Vec::new(),
            };
            let ids = vec![9u32, 4, 4, 8];
            let feeds = vec![feed(1, 2), feed(4, 3)];
            let (r, logits) = serve_cmd(&mut g, Cmd::PrefillMedia(ids.clone(), feeds), 8);
            assert_eq!(r.expect("the call"), 7);
            assert!(logits.is_none());
            assert_eq!(
                g.took,
                vec![(ids, vec![(1, 2u8), (4, 3)])],
                "one call, the ids and both feeds as the seat took them"
            );
            let long = vec![0u32; 1 << 10];
            let e = serve_cmd(&mut g, Cmd::PrefillMedia(long, vec![feed(0, 1)]), 8)
                .0
                .expect_err("a call past the context");
            assert!(
                e.contains("prefill of 1024 ids with 1 image(s) from position 7"),
                "{e}"
            );
            assert_eq!(g.took.len(), 1, "the refused call reached no seat");
        }

        /// The trait's default ([`Seat::prefill_media`]): a call that carries
        /// no image is the seat's plain `prefill`, one that carries any is a
        /// named refusal — the engine that took no [`Seat::media_model`]
        /// never answers an image with text.
        #[test]
        fn the_default_seat_refuses_an_image_by_name() {
            let mut g = NoImages::new();
            let (r, _) = serve_cmd(&mut g, Cmd::PrefillMedia(vec![1, 2], Vec::new()), 8);
            assert_eq!(r.expect("the plain call"), 5);
            assert_eq!(g.took, vec![(vec![1, 2], Vec::new())]);
            let e = serve_cmd(
                &mut g,
                Cmd::PrefillMedia(vec![1, 2, 3], vec![feed(0, 1)]),
                8,
            )
            .0
            .expect_err("an image the default refuses");
            assert!(e.contains("this seat takes no image input"), "{e}");
            assert_eq!(g.took.len(), 1, "the refused call reached no prefill");
        }
    }

    mod slot_files {
        use std::io::{self, Read, Write};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::thread::JoinHandle;

        use serve::{
            CacheNote, EngineError, Flight, Landed, Pending, Saved, SavedState, StateError,
        };

        use bloomery_gpu::GpuError;
        use bloomery_gpu::checkpoint::saved::SeqView;
        use bloomery_gpu::checkpoint::staging::NoRoom;

        use super::super::{
            Cached, Cmd, Extra, Landing, Link, PIPE_CHUNK, PutBack, Reply, Seat, begin, begin_of,
            put_back, restore_from, resume_of, resume_prefix_of, save_into, settle, snapshot_of,
            snapshot_size_of,
        };
        use crate::GateError;

        /// The [`Seat`] methods a slot file's command never calls.
        macro_rules! no_steps {
            () => {
                fn pos(&self) -> usize {
                    self.pos
                }
                fn ctx_max(&self) -> usize {
                    1 << 10
                }
                fn prefill(&mut self, _: &[u32]) -> Result<u32, GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn step(&mut self, _: u32) -> Result<u32, GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn logits_into(&self, _: &mut [f32]) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn reset(&mut self) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn rollback(&mut self, _: u32) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn keep(&self, _: usize) -> (usize, Option<String>) {
                    (0, None)
                }
                fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
                    Vec::new()
                }
                fn note(_: &CacheNote) {}
            };
        }

        /// The prompt-cache methods of a seat that serves slot files alone.
        macro_rules! no_cache {
            () => {
                fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn resume(&mut self, _: &dyn Saved) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
            };
        }

        /// A seat that keeps the trait's slot-file defaults.
        struct NoFiles {
            pos: usize,
        }

        impl Seat for NoFiles {
            no_steps!();
            no_cache!();
        }

        /// The positions [`Files`] stands at after a restore.
        const RESTORED: usize = 5;

        /// A seat whose state is `state`, which a restore replaces with what
        /// it read; `fail` makes both refuse with that kind instead.
        struct Files {
            pos: usize,
            state: Vec<u8>,
            fail: Option<Fail>,
        }

        #[derive(Clone, Copy)]
        enum Fail {
            Format,
            Engine,
        }

        impl Fail {
            fn error(self) -> StateError {
                match self {
                    Fail::Format => StateError::Format("a state of another model".to_owned()),
                    Fail::Engine => StateError::Engine(EngineError("the card fell off".to_owned())),
                }
            }
        }

        impl Seat for Files {
            no_steps!();
            no_cache!();

            fn save_state(&mut self, out: &mut dyn Write) -> Result<u64, StateError> {
                if let Some(f) = self.fail {
                    return Err(f.error());
                }
                out.write_all(&self.state)?;
                Ok(self.state.len() as u64)
            }

            fn restore_state(&mut self, input: &mut dyn Read) -> Result<usize, StateError> {
                if let Some(f) = self.fail {
                    return Err(f.error());
                }
                self.state.clear();
                input.read_to_end(&mut self.state)?;
                self.pos = RESTORED;
                Ok(RESTORED)
            }
        }

        /// A seat whose prompt-cache calls fail with `fail`'s error.
        struct Caches {
            pos: usize,
            fail: fn() -> GateError,
        }

        impl Seat for Caches {
            no_steps!();

            fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
                Err((self.fail)())
            }

            fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
                Err((self.fail)())
            }

            fn resume(&mut self, _: &dyn Saved) -> Result<(), GateError> {
                Err((self.fail)())
            }
        }

        /// A prompt-cache state of the positions it names, which no seat here
        /// takes back.
        struct Held(usize);

        impl Saved for Held {
            fn n_tokens(&self) -> usize {
                self.0
            }

            fn n_bytes(&self) -> u64 {
                0
            }

            fn keepable(&self, n: usize) -> usize {
                n.min(self.0)
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        /// What a landed copy cost.
        const COST: Landed = Landed {
            copy_ms: 1.5,
            calls: 6,
            alloc_ms: 0.0,
        };

        /// A copy that lands on its `at`-th poll; a wait lands it at once.
        struct Slow {
            polls: AtomicUsize,
            at: usize,
        }

        impl Flight for Slow {
            fn poll(&self) -> Result<Option<Landed>, StateError> {
                Ok((self.polls.fetch_add(1, Ordering::SeqCst) + 1 >= self.at).then_some(COST))
            }

            fn wait(&self) -> Result<Landed, StateError> {
                Ok(COST)
            }
        }

        /// A seat that copies its snapshots on a stream of its own and puts
        /// back the prefix it is asked for.
        struct Lazy {
            pos: usize,
            flight: Arc<Slow>,
            resumed: Vec<usize>,
        }

        impl Seat for Lazy {
            no_steps!();
            no_cache!();

            fn snapshots_async(&self) -> bool {
                true
            }

            fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
                Ok(Pending::new(Arc::new(Held(30)), self.flight.clone()))
            }

            fn resume_prefix(&mut self, _: &dyn Saved, n: usize) -> Result<usize, GateError> {
                self.resumed.push(n);
                Ok(n)
            }
        }

        /// A seat that keeps the trait's defaults for a begun copy and a
        /// prefix: its snapshot is taken on the call, its resume is the whole
        /// state's.
        struct Whole {
            pos: usize,
            resumes: usize,
            snapshots: usize,
        }

        impl Seat for Whole {
            no_steps!();

            fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
                Ok(Some(0))
            }

            fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
                self.snapshots += 1;
                Ok(Arc::new(Held(30)))
            }

            fn resume(&mut self, _: &dyn Saved) -> Result<(), GateError> {
                self.resumes += 1;
                Ok(())
            }
        }

        /// An error whose source is the one it wraps, as a seat's own error
        /// wraps the body's.
        #[derive(Debug)]
        struct Wrapped(GpuError);

        impl std::fmt::Display for Wrapped {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "the session refused")
            }
        }

        impl std::error::Error for Wrapped {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }

        /// A link to a thread that answers a seat's state commands as the
        /// engine thread does ([`save_into`], [`restore_from`],
        /// [`snapshot_of`], [`resume_of`]); joining it hands the seat back.
        fn file_link<S: Seat + Send>(mut seat: S) -> (Link, JoinHandle<S>) {
            let (tx, cmds) = mpsc::channel::<Cmd>();
            let (replies, rx) = mpsc::channel::<Reply>();
            let worker = std::thread::spawn(move || {
                for cmd in cmds {
                    let extra = match cmd {
                        Cmd::SaveState(chunks) => Extra::State(save_into(&mut seat, chunks)),
                        Cmd::RestoreState(chunks) => Extra::State(restore_from(&mut seat, chunks)),
                        Cmd::Save => Extra::Saved(snapshot_of(&mut seat)),
                        Cmd::SaveSize => Extra::SaveSize(snapshot_size_of(&mut seat)),
                        Cmd::Resume(state) => Extra::Resumed(resume_of(&mut seat, &*state)),
                        Cmd::SaveBegin => Extra::Begun(begin_of(&mut seat)),
                        Cmd::SavePoll(p) => Extra::Polled(p.flight().poll()),
                        Cmd::SaveWait(p) => Extra::Polled(p.flight().wait().map(Some)),
                        Cmd::ResumePrefix(state, n) => {
                            Extra::Prefix(resume_prefix_of(&mut seat, &*state, n))
                        }
                        _ => panic!("the file thread answers a seat's state commands only"),
                    };
                    let reply = Reply {
                        result: Ok(0),
                        logits: None,
                        pos: seat.pos(),
                        extra,
                    };
                    if replies.send(reply).is_err() {
                        break;
                    }
                }
                seat
            });
            let link = Link {
                tx: Some(tx),
                rx,
                logits: Vec::new(),
                scratch: Vec::new(),
            };
            (link, worker)
        }

        /// A copy begun on the engine thread is asked and waited for there: the
        /// begun state reaches the link at once, a poll answers nothing while
        /// the copy runs and its cost once it has landed, a wait answers the
        /// cost, and a prefix goes back through the seat's own call; a seat
        /// that keeps the defaults takes its snapshot on the call (landed
        /// already, its `snapshots_async` false) and puts the whole state back,
        /// saying so.
        #[test]
        fn a_begun_copy_is_asked_and_waited_for_on_the_engine_thread() {
            let flight = Arc::new(Slow {
                polls: AtomicUsize::new(0),
                at: 2,
            });
            let lazy = Lazy {
                pos: 30,
                flight: Arc::clone(&flight),
                resumed: Vec::new(),
            };
            assert!(lazy.snapshots_async());
            let (mut link, worker) = file_link(lazy);
            let p = link.snapshot_begin().expect("a begun copy");
            assert_eq!(p.state().n_tokens(), 30);
            assert_eq!(
                link.flight(&p, false).expect("a poll"),
                None,
                "still running"
            );
            assert_eq!(
                link.flight(&p, false).expect("a poll"),
                Some(COST),
                "landed"
            );
            assert_eq!(link.flight(&p, true).expect("a wait"), Some(COST));
            assert_eq!(link.resume_prefix(p.state(), 12).expect("a prefix"), 12);
            link.tx = None;
            let seat = worker.join().expect("the file thread");
            assert_eq!(seat.resumed, vec![12]);
            assert_eq!(
                flight.polls.load(Ordering::SeqCst),
                2,
                "a wait is not a poll"
            );

            let whole = Whole {
                pos: 30,
                resumes: 0,
                snapshots: 0,
            };
            assert!(!whole.snapshots_async());
            let (mut link, worker) = file_link(whole);
            let p = link.snapshot_begin().expect("a snapshot on the call");
            assert_eq!(
                link.flight(&p, false).expect("a poll"),
                Some(Landed::default())
            );
            assert_eq!(
                link.resume_prefix(p.state(), 12).expect("a whole resume"),
                30
            );
            link.tx = None;
            assert_eq!(worker.join().expect("the file thread").resumes, 1);
        }

        #[test]
        fn a_snapshot_size_crosses_the_link_without_copying() {
            let (mut link, worker) = file_link(Whole {
                pos: 30,
                resumes: 0,
                snapshots: 0,
            });
            assert_eq!(link.snapshot_size().expect("a size query"), Some(0));
            link.tx = None;
            let seat = worker.join().expect("the file thread");
            assert_eq!(seat.snapshots, 0);
            assert_eq!(seat.pos, 30);
        }

        /// A body's sequence state of `.0` positions, every one a point.
        struct Rows(u32);

        impl SeqView for Rows {
            fn positions(&self) -> u32 {
                self.0
            }

            fn bytes(&self) -> usize {
                self.0 as usize * 8
            }

            fn keep_point(&self, n: u32) -> u32 {
                n.min(self.0)
            }
        }

        /// A copy on a lane that has landed once `ready` is set; `lands`
        /// counts the waits for it.
        struct Copy {
            ready: AtomicBool,
            lands: AtomicUsize,
        }

        impl Landing for Copy {
            fn ready(&self) -> Result<bool, GpuError> {
                Ok(self.ready.load(Ordering::SeqCst))
            }

            fn land(&self) -> Result<Landed, GpuError> {
                self.lands.fetch_add(1, Ordering::SeqCst);
                self.ready.store(true, Ordering::SeqCst);
                Ok(COST)
            }
        }

        fn copy() -> Arc<Copy> {
            Arc::new(Copy {
                ready: AtomicBool::new(false),
                lands: AtomicUsize::new(0),
            })
        }

        /// A seat whose staging pool has no room for the state it was asked
        /// to begin, and says so as the body's [`NoRoom`] under its own error.
        struct Short {
            pos: usize,
            wrapped: bool,
        }

        impl Seat for Short {
            no_steps!();
            no_cache!();

            fn snapshots_async(&self) -> bool {
                true
            }

            fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
                let short = NoRoom {
                    need: 4096,
                    held: 8192,
                    lent: 6144,
                };
                if self.wrapped {
                    return Err(Box::new(Pooled(short)));
                }
                begin::<Rows, ()>(Err(short), || Ok(()))
            }
        }

        /// An error whose source is a pool's refusal, as a seat's own error
        /// wraps the body's.
        #[derive(Debug)]
        struct Pooled(NoRoom);

        impl std::fmt::Display for Pooled {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "the save failed")
            }
        }

        impl std::error::Error for Pooled {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }

        /// A pool with no room for a save is the server's `NoRoom` with the
        /// pool's three numbers (which it answers by evicting and asking
        /// again), whether the seat hands the refusal up as it is or wrapped
        /// in its own error — never `Engine`, which ends the server.
        #[test]
        fn a_full_pool_is_the_servers_no_room_not_an_engine_failure() {
            for wrapped in [false, true] {
                let (mut link, worker) = file_link(Short { pos: 30, wrapped });
                let Err(e) = link.snapshot_begin() else {
                    panic!("wrapped {wrapped}: a pool with no room began a copy");
                };
                match e {
                    StateError::NoRoom { need, held, lent } => {
                        assert_eq!((need, held, lent), (4096, 8192, 6144), "wrapped {wrapped}");
                    }
                    other => panic!("wrapped {wrapped}: {other:?} is not NoRoom"),
                }
                assert_eq!(e.fallback(), Some("full"));
                link.tx = None;
                worker.join().expect("the file thread");
            }
        }

        /// The body's state of a begun copy is read only after the copy has
        /// landed: [`Cached::body`] waits for it, once for every reader, and
        /// the server's flight answers nothing while the copy runs and its
        /// cost after. A state taken on the request path waits for nothing.
        #[test]
        fn a_cached_state_is_read_only_after_its_copy_landed() {
            let lane = copy();
            let landing: Arc<dyn Landing> = lane.clone();
            let p = begin(Ok((Rows(30), landing)), || Ok(7_u8)).expect("a begun state");
            assert_eq!(p.state().n_tokens(), 30);
            assert_eq!(p.state().n_bytes(), 240);
            assert_eq!(p.state().keepable(12), 12);
            assert_eq!(p.flight().poll().expect("a poll"), None, "still running");
            assert_eq!(lane.lands.load(Ordering::SeqCst), 0, "a poll does not wait");

            let saved = Cached::<Rows, u8>::of(&**p.state(), "a test body's").expect("this seat's");
            assert_eq!(saved.draft, 7);
            assert_eq!(saved.body().expect("the body").positions(), 30);
            assert_eq!(
                lane.lands.load(Ordering::SeqCst),
                1,
                "the body is read before its copy landed"
            );
            assert_eq!(p.flight().poll().expect("a poll"), Some(COST));
            assert_eq!(p.flight().wait().expect("a wait"), COST);

            let Err(e) = Cached::<Rows, u16>::of(&**p.state(), "a test body's") else {
                panic!("a state of another seat's draft side was taken");
            };
            assert!(e.to_string().contains("not a test body's"), "{e}");

            let own = Cached::landed(Rows(5), ());
            assert_eq!(own.body().expect("the body").positions(), 5);
        }

        /// A seat whose draft steps [`put_back`] drives, each recorded; one
        /// of them may refuse.
        struct Steps {
            calls: Vec<&'static str>,
            refuse: Option<&'static str>,
        }

        impl Steps {
            fn step(&mut self, name: &'static str) -> Result<(), GateError> {
                self.calls.push(name);
                match self.refuse {
                    Some(r) if r == name => Err(format!("{name} refused").into()),
                    _ => Ok(()),
                }
            }
        }

        impl PutBack for Steps {
            type Body = Rows;
            type Side = u8;

            fn takes(&self, _: &u8) -> Result<(), GateError> {
                match self.refuse {
                    Some("takes") => Err("takes refused".into()),
                    _ => Ok(()),
                }
            }

            fn reset_selected(&mut self) -> Result<(), GateError> {
                self.step("reset")
            }

            fn unpark(&mut self, _: &u8) -> Result<(), GateError> {
                self.step("unpark")
            }
        }

        /// A put-back is the side's check, the reset, the body's restore on
        /// the copy that landed, then the side put back, in that order; a
        /// state of another seat, a side refused or a body refused stops it
        /// before the next step, and the unpark is the last step of every
        /// whole run.
        #[test]
        fn a_put_back_runs_its_steps_in_order_and_stops_at_a_refusal() {
            let lane = copy();
            let landing: Arc<dyn Landing> = lane.clone();
            let p = begin(Ok((Rows(30), landing)), || Ok(7_u8)).expect("a begun state");
            let run = |refuse: Option<&'static str>, put: Result<u32, GpuError>| {
                let mut seat = Steps {
                    calls: Vec::new(),
                    refuse,
                };
                let r = put_back(&mut seat, &**p.state(), "a test body's", |s, body| {
                    assert_eq!(body.positions(), 30);
                    s.calls.push("put");
                    put
                });
                (r, seat.calls)
            };
            let (r, calls) = run(None, Ok(12));
            assert_eq!(r.expect("a put-back"), 12);
            assert_eq!(calls, ["reset", "put", "unpark"]);
            assert_eq!(
                lane.lands.load(Ordering::SeqCst),
                1,
                "the copy is waited for"
            );

            let (r, calls) = run(Some("takes"), Ok(12));
            assert!(r.is_err());
            assert!(calls.is_empty(), "a refused side reached {calls:?}");

            let (r, calls) = run(Some("reset"), Ok(12));
            assert!(r.is_err());
            assert_eq!(calls, ["reset"], "a refused reset reached the restore");

            let (r, calls) = run(
                None,
                Err(GpuError::Shape {
                    what: "test",
                    detail: "no".to_owned(),
                }),
            );
            assert!(r.is_err());
            assert_eq!(
                calls,
                ["reset", "put"],
                "a refused restore reached the unpark"
            );

            let (r, calls) = run(Some("unpark"), Ok(12));
            assert!(r.is_err());
            assert_eq!(calls, ["reset", "put", "unpark"]);

            let mut seat = Steps {
                calls: Vec::new(),
                refuse: None,
            };
            let other = Cached::landed(Rows(5), 1_u16);
            let Err(e) = put_back(&mut seat, &other, "a test body's", |_, _| Ok(5)) else {
                panic!("a state of another seat's side was put back");
            };
            assert!(e.to_string().contains("not a test body's"), "{e}");
            assert!(seat.calls.is_empty(), "{:?}", seat.calls);
        }

        /// `n` bytes of a pattern that repeats past no chunk boundary.
        fn pattern(n: usize, salt: u8) -> Vec<u8> {
            (0..n).map(|i| (i % 251) as u8 ^ salt).collect()
        }

        /// A writer whose every write fails, as a full disk does.
        struct Full;

        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("no space left on the device"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        /// A state of several chunks and a partial one crosses the pipe byte
        /// for byte, each way: the seat's bytes reach the server's `out`, the
        /// file's bytes reach the seat, and the answers count them.
        #[test]
        fn a_slot_state_crosses_the_pipe_byte_for_byte() {
            let n = 3 * PIPE_CHUNK + 17;
            let state = pattern(n, 0);
            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: state.clone(),
                fail: None,
            });
            let mut out = Vec::new();
            let saved = link.save_state(&mut out).expect("a save");
            assert_eq!(
                saved,
                SavedState {
                    n_tokens: 9,
                    n_bytes: n as u64
                }
            );
            assert!(out == state, "the saved bytes differ from the seat's");
            let file = pattern(n, 0x5a);
            let restored = link.restore_state(&mut file.as_slice()).expect("a restore");
            assert_eq!(
                restored,
                SavedState {
                    n_tokens: RESTORED,
                    n_bytes: n as u64
                }
            );
            link.tx = None;
            let seat = worker.join().expect("the file thread");
            assert!(
                seat.state == file,
                "the seat read other bytes than the file's"
            );
        }

        /// A seat that keeps the trait's defaults answers `Unsupported` both
        /// ways (the server's 501), and a save writes nothing.
        #[test]
        fn the_default_seat_answers_unsupported() {
            let (mut link, worker) = file_link(NoFiles { pos: 3 });
            let mut out = Vec::new();
            let e = link.save_state(&mut out).expect_err("the default save");
            assert!(
                matches!(e, StateError::Unsupported("slot save/restore")),
                "{e:?}"
            );
            assert!(out.is_empty(), "the default save wrote {} bytes", out.len());
            let file = pattern(PIPE_CHUNK + 1, 0);
            let e = link
                .restore_state(&mut file.as_slice())
                .expect_err("the default restore");
            assert!(
                matches!(e, StateError::Unsupported("slot save/restore")),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");
        }

        /// The kind crosses the thread: a refusal stays `Format` (its words
        /// as the seat gave them), an engine failure stays `Engine` (with the
        /// command and the position), and an error of the server's `out` is
        /// `Io`. An engine failure is never hidden behind the stream's error.
        #[test]
        fn a_state_error_keeps_its_kind() {
            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: Vec::new(),
                fail: Some(Fail::Format),
            });
            let e = link
                .restore_state(&mut [1u8, 2, 3].as_slice())
                .expect_err("a refused restore");
            assert!(
                matches!(&e, StateError::Format(m) if m == "a state of another model"),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");

            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: Vec::new(),
                fail: Some(Fail::Engine),
            });
            let e = link.save_state(&mut Vec::new()).expect_err("a failed save");
            assert!(
                matches!(&e, StateError::Engine(EngineError(m))
                    if m == "save at position 9: the card fell off"),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");

            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: pattern(3 * PIPE_CHUNK, 0),
                fail: None,
            });
            let e = link
                .save_state(&mut Full)
                .expect_err("a save to a full disk");
            assert!(
                matches!(&e, StateError::Io(io) if io.to_string() == "no space left on the device"),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");

            let stream = || Err(io::Error::other("no space left on the device"));
            let engine = Err(StateError::Engine(EngineError(
                "the card fell off".to_owned(),
            )));
            assert!(matches!(
                settle(engine, stream()),
                Err(StateError::Engine(_))
            ));
            let format = Err(StateError::Format("a broken pipe".to_owned()));
            assert!(matches!(settle(format, stream()), Err(StateError::Io(_))));
        }

        /// A prompt-cache state the seat refuses by name — a `GpuError` of
        /// kind `Shape` or `State`, itself or in the chain — reaches the
        /// server as `Format`, the request's (the server keeps no state or
        /// resets the slot), not `Engine`, which would end it. Any other
        /// failure stays the engine's.
        #[test]
        fn a_refused_cache_state_is_the_requests() {
            fn waits() -> GateError {
                Box::new(GpuError::Shape {
                    what: "deepseek41 sequence state",
                    detail: "a call while the pass of slots [0, 1] waits for its commit".to_owned(),
                })
            }
            fn missing() -> GateError {
                Box::new(Wrapped(GpuError::State {
                    what: "deepseek41 sequence state",
                    missing: "a reset",
                }))
            }
            fn lost() -> GateError {
                Box::new(GpuError::Protocol {
                    what: "the host tier",
                    detail: "a go that did not land".to_owned(),
                })
            }
            fn unnamed() -> GateError {
                "a saved state that is not this body's".into()
            }
            let held: Arc<dyn Saved> = Arc::new(Held(4));
            let cases: [(fn() -> GateError, bool); 4] = [
                (waits, true),
                (missing, true),
                (lost, false),
                (unnamed, false),
            ];
            for (fail, refused) in cases {
                let (mut link, worker) = file_link(Caches { pos: 7, fail });
                let size = link.snapshot_size().expect_err("a failed size query");
                let saved = link.snapshot().map(|_| ()).expect_err("a failed snapshot");
                let resumed = link.resume(&held).expect_err("a failed resume");
                let begun = link
                    .snapshot_begin()
                    .map(|_| ())
                    .expect_err("a failed begin");
                let prefix = link.resume_prefix(&held, 3).expect_err("a failed prefix");
                let want = fail().to_string();
                for (e, head) in [
                    (&size, "snapshot size at position 7: "),
                    (&saved, "snapshot at position 7: "),
                    (&resumed, "resume of a state of 4 positions: "),
                    (&begun, "snapshot begun at position 7: "),
                    (
                        &prefix,
                        "resume of the first 3 positions of a state of 4 positions: ",
                    ),
                ] {
                    let m = match e {
                        StateError::Format(m) if refused => m,
                        StateError::Engine(EngineError(m)) if !refused => m,
                        other => panic!("{want}: {other:?}"),
                    };
                    assert_eq!(*m, format!("{head}{want}"));
                }
                link.tx = None;
                worker.join().expect("the file thread");
            }
        }
    }

    mod cache_ram {
        use super::super::{CacheRam, HostRead};
        use crate::generate::Place;
        use crate::residency38::{
            CHECKPOINTS_RESERVE, Seqs, check_reserve, checkpoints_pinned, checkpoints_reserved,
            glm_machine, reserve_checkpoints,
        };
        use model::placement::{Host, Machine, Node};
        use runtime::seqstate::{HOST_BUDGET, Price};

        /// One GLM checkpoint's bytes: 34 KDA layers' recurrent state and
        /// conv ring (5,275,648 B a layer).
        const GLM_POINT: u64 = 34 * 5_275_648;
        /// The spacing of a GLM prompt call's inner checkpoints.
        const GLM_EVERY: u32 = 512;
        /// GLM's price: its point and its spacing.
        const GLM_PRICE: Price = Price {
            point_bytes: GLM_POINT,
            every: GLM_EVERY,
        };
        /// A price whose slot of 4,096 positions takes the four points of 1 GiB
        /// the host budget fits: 4 GiB a slot.
        const FOUR_GIB: Price = Price {
            point_bytes: 1 << 30,
            every: 512,
        };

        /// A machine of `os` reserved bytes on the host and nothing else.
        fn machine(os: u64) -> Machine {
            Machine {
                cards: Vec::new(),
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: 0,
                        reserves: vec![("os".to_owned(), os)],
                    },
                    unified: None,
                }],
            }
        }

        /// The default budget's zero is a named record, not a silent one: a
        /// reading that leaves nothing past the need and the pool gives
        /// `ram=0` with a `why=` naming the arithmetic, and the `cache` line
        /// carries it; a leftover halves under the cap with no why, and one
        /// byte left halves to a named zero too.
        #[test]
        fn a_default_budget_of_zero_names_its_terms() {
            let (ram, why) = CacheRam::default_of(10_000, 6_000, 1_000);
            assert_eq!((ram, why.is_none()), (1_500, true));
            for available in [6_000 + 1_000, 6_000 + 1_000 + 1] {
                let (ram, why) = CacheRam::default_of(available, 6_000, 1_000);
                assert_eq!(ram, 0, "nothing left at {available} B");
                let why = why.expect("the zero is named");
                for part in [
                    format!("the reading {available} B leaves"),
                    "past need 6000 B + pool 1000 B".to_owned(),
                    "no prompt cache".to_owned(),
                ] {
                    assert!(why.contains(&part), "{part:?} in {why}");
                }
            }
            let (_, why) = CacheRam::default_of(6_000, 6_000, 1_000);
            let line = CacheRam {
                ram: 0,
                set: false,
                available: 6_000,
                reading: HostRead::Given,
                need: 6_000,
                pool: 1_000,
                tier: 0,
                paged: 0,
                checkpoints: 0,
                why,
            }
            .line();
            assert!(line.contains("ram=0"), "{line}");
            assert!(line.contains(" why=the reading 6000 B leaves"), "{line}");
        }

        /// A plan the NVMe tier split takes no default budget: the split
        /// fills the room past the tier's floor (its arena and the host
        /// experts it keeps, inside the need), so a reading past the need is
        /// rounding or drift — on a split with an arena and on one without
        /// (the mapping path), the budget is 0 and `why=` names the split, and
        /// the line prints the arena as `tier` and the drive's bytes as
        /// `paged`. A set budget keeps its value. A need under the arena it
        /// holds, read without it, and an arena on a plan that pages nothing
        /// are refused by name.
        #[test]
        fn a_split_plan_takes_no_default_budget() {
            let checkpoints =
                checkpoints_pinned(Seqs::priced(1, FOUR_GIB), 4_096).expect("a reserve");
            let (plan, pool, arena, paged) = (6_000u64, 1_000u64, 2_000u64, 3_000u64);
            let available = 100_000 + checkpoints;
            for (tier, paged) in [(arena, paged), (0, paged)] {
                let c = CacheRam::at(
                    None,
                    (available, HostRead::Given),
                    plan + tier + checkpoints,
                    pool,
                    (tier, paged),
                    checkpoints,
                )
                .expect("the budget");
                assert_eq!((c.ram, c.tier, c.paged), (0, tier, paged));
                let why = c.why.as_deref().expect("the zero is named");
                for part in [
                    "the NVMe tier's split fills the room past its floor".to_owned(),
                    format!("arena {tier} B"),
                    format!("{paged} B read from the drive"),
                    "no prompt cache".to_owned(),
                ] {
                    assert!(why.contains(&part), "{part:?} in {why}");
                }
                let line = c.line();
                assert!(
                    line.contains(&format!(" tier={tier} paged={paged} ")),
                    "{line}"
                );
                let set = CacheRam::at(
                    Some(4_096),
                    (available, HostRead::Given),
                    plan + tier + checkpoints,
                    pool,
                    (tier, paged),
                    checkpoints,
                )
                .expect("the set budget");
                assert_eq!((set.ram, set.why.as_deref()), (4_096, None));
            }
            let Err(e) = CacheRam::at(
                None,
                (available, HostRead::Given),
                plan - 5_000,
                pool,
                (arena, paged),
                checkpoints,
            ) else {
                panic!("a need without its arena is refused");
            };
            let e = e.to_string();
            assert!(e.contains("under the NVMe tier arena 2000 B"), "{e}");
            let Err(e) = CacheRam::at(
                None,
                (available, HostRead::Given),
                plan + arena,
                pool,
                (arena, 0),
                checkpoints,
            ) else {
                panic!("an arena on a plan that pages nothing is refused");
            };
            let e = e.to_string();
            assert!(
                e.contains("arena of 2000 B and pages no routed expert"),
                "{e}"
            );
        }

        /// A default budget too small for one saved state is off: `holding`
        /// at one byte over it gives 0 with a `why=` naming both, at the
        /// state's own bytes it keeps the budget, a zero stays the zero it
        /// was, and a set budget keeps its value at any state size.
        #[test]
        fn a_default_under_one_saved_state_is_off() {
            let (need, pool) = (6_000u64, 1_000u64);
            let available = need + pool + 100_000;
            let at = |set: Option<u64>| {
                CacheRam::at(set, (available, HostRead::Given), need, pool, (0, 0), 0)
                    .expect("the budget")
            };
            assert_eq!(at(None).ram, 50_000);
            let off = at(None).holding(50_001);
            assert_eq!(off.ram, 0);
            let why = off.why.as_deref().expect("the zero is named");
            assert!(
                why.contains("the default budget 50000 B is under one saved state's 50001 B"),
                "{why}"
            );
            assert!(off.line().contains("ram=0 rule=default"), "{}", off.line());
            let kept = at(None).holding(50_000);
            assert_eq!((kept.ram, kept.why.as_deref()), (50_000, None));
            let set = at(Some(10)).holding(11);
            assert_eq!((set.ram, set.set, set.why.as_deref()), (10, true, None));
            let none = CacheRam::at(None, (need, HostRead::Given), need, pool, (0, 0), 0)
                .expect("the budget");
            let why = none.why.clone();
            let still = none.holding(u64::MAX);
            assert_eq!((still.ram, still.why), (0, why));
        }

        /// A budget under the one session's state the load serves at is kept
        /// and says so: `short_of` at one byte over the budget gives the same
        /// budget with a `why=` naming both and the context, at the state's
        /// own bytes (and for a budget of 0, which has its own `why`) it
        /// changes nothing, and a set budget keeps its value with the `why`.
        #[test]
        fn a_budget_under_one_session_is_kept_and_named() {
            let (need, pool) = (6_000u64, 1_000u64);
            let available = need + pool + 100_000;
            let at = |set: Option<u64>| {
                CacheRam::at(set, (available, HostRead::Given), need, pool, (0, 0), 0)
                    .expect("the budget")
            };
            let short = at(None).short_of(4096, 50_001);
            assert_eq!(short.ram, 50_000, "the budget stays on");
            let why = short.why.as_deref().expect("the shortfall is named");
            assert!(
                why.contains("budget 50000 B is under one session's state of 50001 B at the 4096"),
                "{why}"
            );
            assert!(short.line().contains(" why="), "{}", short.line());
            let held = at(None).short_of(4096, 50_000);
            assert_eq!((held.ram, held.why.as_deref()), (50_000, None));
            let set = at(Some(10)).short_of(4096, 11);
            assert_eq!((set.ram, set.set), (10, true));
            assert!(set.why.is_some(), "a set budget is named too");
            let none = CacheRam::at(None, (need, HostRead::Given), need, pool, (0, 0), 0)
                .expect("the budget");
            let why = none.why.clone();
            let still = none.short_of(4096, u64::MAX);
            assert_eq!((still.ram, still.why), (0, why));
        }

        /// The default budget counts a load's checkpoint reserve once: the
        /// plan's need holds the machine's `checkpoints` row, so the room
        /// past the need is the reading less the plan's own terms less that
        /// row, the budget half of it, and `checkpoints` is the term the
        /// `cache` line prints, not a second subtraction. One example slot
        /// at a context of 16,384 positions pins five points of 148,065,344
        /// B spaced 4,096 apart (740,326,720 B) inside the need of
        /// 27,721,043,264 B: the same plan with no reserve leaves that much
        /// more room, and half of it more budget. The line's own terms
        /// (`available`, `need`, `pool`) reproduce the budget it prints.
        #[test]
        fn a_default_budget_counts_the_checkpoint_reserve_once() {
            const EXAMPLE: Price = Price {
                point_bytes: 148_065_344,
                every: 4_096,
            };
            let mut m = machine(0);
            reserve_checkpoints(&mut m, Seqs::priced(1, EXAMPLE), 16_384).expect("a reserve");
            let reserved = checkpoints_reserved(&m);
            assert_eq!(reserved, 5 * 148_065_344);
            let (terms, pool, available) = (26_980_716_544u64, 0u64, 40_000_000_000u64);
            let need = terms + reserved;
            assert_eq!(need, 27_721_043_264);
            let at = |need: u64, checkpoints: u64| {
                CacheRam::at(
                    None,
                    (available, HostRead::Given),
                    need,
                    pool,
                    (0, 0),
                    checkpoints,
                )
                .expect("the budget")
            };
            let (with, without) = (at(need, reserved), at(terms, 0));
            assert_eq!(with.ram, (available - need - pool) / 2);
            assert_eq!(without.ram, (available - terms - pool) / 2);
            assert_eq!(without.ram - with.ram, reserved / 2);
            assert_eq!(with.why, None);
            let line = with.line();
            let term = |key: &str| -> u64 {
                line.split(' ')
                    .find_map(|w| w.strip_prefix(&format!("{key}=")))
                    .unwrap_or_else(|| panic!("{key} in {line}"))
                    .parse()
                    .unwrap_or_else(|_| panic!("{key} is a number in {line}"))
            };
            assert_eq!(term("checkpoints"), reserved, "{line}");
            assert_eq!(term("need"), need, "{line}");
            assert_eq!(
                term("ram"),
                (term("available") - term("need") - term("pool")) / 2,
                "{line}"
            );
        }

        /// The checkpoint reserve is each load's declaration
        /// (`reserve_checkpoints`): V4.1's body makes no checkpoints, so its
        /// seat declares none and its machine carries no row and reserves 0;
        /// the Qwen3.8 seat's two slots, taking them, carry one row of the
        /// price's budget at the slot's context a slot; a second declaration on
        /// one machine is a panic by name.
        #[test]
        fn the_checkpoint_reserve_is_each_loads_declaration() {
            let declared = |seqs: Seqs| {
                let mut m = machine(1_000);
                reserve_checkpoints(&mut m, seqs, 4_096).expect("a reserve");
                m
            };
            let v41 = declared(Seqs::none(2));
            assert_eq!(v41.host().reserves, vec![("os".to_owned(), 1_000)]);
            assert_eq!(checkpoints_reserved(&v41), 0);
            let slot = FOUR_GIB.budget(4_096).expect("a budget");
            let q38 = declared(Seqs::priced(2, FOUR_GIB));
            assert_eq!(
                q38.host().reserves,
                vec![
                    ("os".to_owned(), 1_000),
                    (CHECKPOINTS_RESERVE.to_owned(), 2 * slot)
                ]
            );
            assert_eq!(checkpoints_reserved(&q38), 2 * slot);
            let twice = std::panic::catch_unwind(|| {
                let mut m = declared(Seqs::priced(2, FOUR_GIB));
                reserve_checkpoints(&mut m, Seqs::ONE, 4_096).expect("a reserve");
            })
            .expect_err("a second declaration panics");
            let said = twice
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| twice.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            assert!(said.contains("reserves its checkpoints already"), "{said}");
        }

        /// A qwen35moe or Qwen3.8 machine's checkpoint row at a total context
        /// `C` split across `N` slots is `N` of the family's budget at a
        /// slot's `C / N` positions: the family's price is its point and its
        /// spacing (`checkpoint_price35`, `checkpoint_price38`, pinned where
        /// they are made: `gate-gpu-lib`) and the budget one sequence's
        /// `Checkpoints` is made with is the price's at the slot's context.
        /// The example prices of 73,728,000 B and 148,065,344 B, spaced
        /// 4,096 apart, take two points in a slot
        /// of 2,048 positions and four in 10,240 (a whole call's points and
        /// the previous call's end) — not the 4 GiB of the default a slot.
        #[test]
        fn a_qwen_machine_reserves_its_familys_budget_at_each_context() {
            for point_bytes in [73_728_000u64, 148_065_344] {
                let price = Price {
                    point_bytes,
                    every: 4_096,
                };
                for (total, slots, points) in [
                    (4_096u64, 2usize, 2u64),
                    (20_480, 2, 4),
                    (10_240, 1, 4),
                    (81_920, 4, 6),
                ] {
                    let slot = total / slots as u64;
                    let mut m = machine(1_000);
                    reserve_checkpoints(&mut m, Seqs::priced(slots, price), slot)
                        .expect("a reserve");
                    let each = price.budget(slot).expect("a budget");
                    assert_eq!(each, points * point_bytes, "{total} over {slots}");
                    assert_eq!(
                        checkpoints_reserved(&m),
                        slots as u64 * each,
                        "{total} over {slots}"
                    );
                    assert_ne!(checkpoints_reserved(&m), slots as u64 * HOST_BUDGET);
                }
            }
        }

        /// A deeper probe never reserves less: the machine's checkpoint row
        /// at each context a search can probe holds at the row of every
        /// shallower one, for an example recurrent price and GLM's, so a search's probes
        /// never reserve under the load they bound.
        #[test]
        fn a_deeper_probe_never_reserves_less() {
            let recurrent = Price {
                point_bytes: 148_065_344,
                every: 4_096,
            };
            for price in [recurrent, GLM_PRICE] {
                let mut last = 0;
                for ctx in (1..=140_000u64).step_by(7) {
                    let mut m = machine(1_000);
                    reserve_checkpoints(&mut m, Seqs::priced(2, price), ctx).expect("a reserve");
                    let reserved = checkpoints_reserved(&m);
                    assert!(reserved >= last, "ctx {ctx}: {reserved} B under {last} B");
                    last = reserved;
                }
            }
        }

        /// A GLM load's machine (`glm_machine`, the one its offer, plans and
        /// open take) reserves every slot's checkpoints at the body's budget
        /// for the load's context: at 4,096 positions that is the 9 points
        /// one prompt call over it takes (the marks 512 to 3,584 and its
        /// end) and the previous call's end, not the 4 GiB of the default, one budget at one slot and
        /// three at three. At one slot the prompt cache's default budget is
        /// the one the load's single sequence left when the term sat beside
        /// the need (half of the reading past need, pool and one budget); at
        /// three the two further slots' come off it too.
        #[test]
        fn a_glm_load_reserves_every_slots_checkpoints() {
            let layers = 4;
            let (need, pool) = (6_000u64, 1_000u64);
            let available = need + pool + 4 * HOST_BUDGET + 100_000;
            for slots in [1usize, 3] {
                let slots_u64 = u64::try_from(slots).expect("a slot count fits u64");
                let m = glm_machine(Place::A, None, slots, GLM_PRICE, 4_096)
                    .expect("plan (a)'s machine")(layers);
                let reserved = checkpoints_reserved(&m);
                assert_eq!(reserved, slots_u64 * 9 * GLM_POINT, "{slots} slots");
                assert_ne!(reserved, slots_u64 * HOST_BUDGET, "{slots} slots");
                let c = CacheRam::at(
                    None,
                    (available, HostRead::Given),
                    need + reserved,
                    pool,
                    (0, 0),
                    reserved,
                )
                .expect("the budget");
                let beside_one = (available - need - pool - 9 * GLM_POINT) / 2;
                assert_eq!(
                    c.ram,
                    beside_one - (slots_u64 - 1) * 9 * GLM_POINT / 2,
                    "{slots} slots"
                );
            }
        }

        /// A GLM load's reserve follows the context it is granted: a slot
        /// holds the points one prompt call over its context takes and the
        /// previous call's end, 2 at 512 positions, 9 at 4,096, and past
        /// 10,752 positions the 23 that
        /// `HOST_BUDGET` fits — less than `HOST_BUDGET` itself, which is not a
        /// multiple of a checkpoint. A second declaration on a machine, by
        /// either owner, is a panic by name.
        #[test]
        fn a_glm_reserve_follows_the_granted_context() {
            for (ctx, points) in [(512u64, 2u64), (4_096, 9), (10_753, 23), (16_384, 23)] {
                let m =
                    glm_machine(Place::A, None, 2, GLM_PRICE, ctx).expect("plan (a)'s machine")(4);
                assert_eq!(
                    checkpoints_reserved(&m),
                    2 * points * GLM_POINT,
                    "ctx {ctx}"
                );
            }
            assert_eq!(23 * GLM_POINT, 4_125_556_736);
            const { assert!(23 * GLM_POINT < HOST_BUDGET) };
            let twice = std::panic::catch_unwind(|| {
                let mut m = machine(1_000);
                reserve_checkpoints(&mut m, Seqs::priced(1, GLM_PRICE), 4_096).expect("a reserve");
                reserve_checkpoints(&mut m, Seqs::ONE, 4_096).expect("a reserve");
            })
            .expect_err("a second declaration panics");
            let said = twice
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| twice.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            assert!(said.contains("reserves its checkpoints already"), "{said}");
        }

        /// The plan's reserve must cover what the load's sequences pin: a
        /// machine reserved for a context of 512 positions does not cover a
        /// load of 16,384, refused by name with both figures; the machine
        /// reserved for the load's own context covers it, and a deeper one
        /// does too.
        #[test]
        fn a_glm_reserve_short_of_the_load_is_refused_by_name() {
            let at = |ctx: u64| GLM_PRICE.budget(ctx).expect("a budget");
            let load = at(16_384);
            let reserved = |ctx: u64| {
                glm_machine(Place::A, None, 2, GLM_PRICE, ctx).expect("plan (a)'s machine")(4)
            };
            let Err(e) = check_reserve(&reserved(512), 2, load) else {
                panic!("a reserve of 512 positions covers a load of 16,384");
            };
            let e = e.to_string();
            assert!(
                e.contains(&format!("reserves {} B of checkpoints", 2 * 2 * GLM_POINT))
                    && e.contains(&format!("pin up to {} B", 2 * 23 * GLM_POINT)),
                "{e}"
            );
            assert!(check_reserve(&reserved(16_384), 2, load).is_ok());
            assert!(check_reserve(&reserved(16_384), 2, at(4_096)).is_ok());
        }

        /// The default budget reads the checkpoints once, inside the need:
        /// a machine that reserves two slots' has them in the plan's need,
        /// so the budget is half of what the reading leaves past that need
        /// and the pool alone, the line printing the reserve as
        /// `checkpoints`; a need under the checkpoints it says it holds (a
        /// need of a machine with no reserve) is refused by name.
        #[test]
        fn a_default_budget_takes_the_reserved_checkpoints_once() {
            let mut m = machine(1_000);
            reserve_checkpoints(&mut m, Seqs::priced(2, FOUR_GIB), 4_096).expect("a reserve");
            let checkpoints = checkpoints_reserved(&m);
            let (need, pool) = (6_000 + checkpoints, 1_000u64);
            let available = need + pool + 100_000;
            let c = CacheRam::at(
                None,
                (available, HostRead::Given),
                need,
                pool,
                (0, 0),
                checkpoints,
            )
            .expect("the budget");
            assert_eq!((c.ram, c.why.as_deref()), (50_000, None));
            assert!(
                c.line().ends_with(&format!(" checkpoints={checkpoints}")),
                "{}",
                c.line()
            );
            let Err(e) = CacheRam::at(
                None,
                (available, HostRead::Given),
                6_000,
                pool,
                (0, 0),
                checkpoints,
            ) else {
                panic!("a need without its machine's checkpoints is refused");
            };
            let e = e.to_string();
            assert!(
                e.contains(&format!("and the checkpoints {checkpoints} B it holds")),
                "{e}"
            );
        }
    }
}
