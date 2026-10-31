//! Routes and JSON shapes. The field names, defaults and stream framing follow
//! llama-server (ik_llama.cpp `examples/server`) so its clients work unchanged.
//!
//! N slots (`--parallel N`, [`SlotConfig`]): the engine runs on an engine
//! thread of its own ([`crate::worker`]), and an HTTP thread hands it a
//! request through the board ([`crate::sched`]), which gives the requests
//! slots in arrival order, then waits for the request's events on a channel.
//! A request that would wait past the queue's depth is a 503 with
//! `Retry-After` at once. `/health`, `/props`, `/slots`, `/metrics`,
//! `/v1/models` and the tokenizer endpoints never wait for the engine.
//!
//! No request takes the engine past [`Engine::ctx_max`], as llama-server's
//! slots do not pass `n_ctx`: a prompt of `ctx_max` tokens or more is a 400
//! before it reaches the engine, naming the one `--ctx-size` that would hold
//! it ([`ctx_hint`]), and generation stops at `ctx_max` with `truncated`,
//! whatever `max_tokens` asked. Every slot has the whole `ctx_max`; it is not
//! split among them.
//!
//! At most [`MAX_CONNECTIONS`] connections are served at once, one thread each;
//! one connection past them gets a 503 with `Retry-After` on a separate thread;
//! further arrivals close at once while that refusal is being written. A
//! connection that sends nothing of a next request for [`KEEP_ALIVE_IDLE`] is
//! closed, so an idle client gives its place back. An `accept` that fails
//! while the listener still stands is a named line on stderr and the loop goes
//! on; a listener that no longer stands ends [`Server::run`].
//!
//! An engine error is fatal: the requests that met it get a 500 carrying the
//! engine's message, every later generation and `/health` a 503 with the reason,
//! and after [`ServerConfig::fatal_linger`] [`Server::run`] returns
//! [`ServeError::Engine`] so the process exits instead of holding the port with a
//! dead engine behind it.
//!
//! `POST /slots/{id}?action=save|restore|erase` answers as llama-server's
//! handlers do, with one difference: an action on a slot that is running a
//! request, or while a request waits for a slot, is a 503 at once where
//! llama-server defers it until the slot is free. Save and restore take
//! `{"filename": "<base name>"}` under [`ServerConfig::slot_save_path`]; the
//! file is [`crate::slotfile`]'s.
//!
//! A field sent with a type it does not take is a 400 naming the field and the
//! type it takes, never read as unset (`null` is the one unset spelling beside
//! absence; the typed getters `get_i`, `get_f32`, `get_b` and `stop_list` own
//! the reads): llama-server falls back to the field's default on a wrong type
//! in its OpenAI pre-parse and ignores a `stop` that is neither a string nor
//! an array, and this server does neither.
//!
//! `POST /residency/reset` is bloomery's own: the engine's adaptive expert
//! residency back to its load's placement ([`Engine::residency_reset`]), the
//! one call that does it — a request never resets it. It runs while every slot
//! is free and no request waits.
//!
//! The server stops the orderly way llama-server does: the first `SIGINT` or
//! `SIGTERM`, or a loopback `POST /shutdown` (bloomery's own), stops admitting
//! work, ends the engine thread between its engine calls, and exits 0 after
//! one stderr line naming the cause; a second signal terminates at once. The
//! stop's one owner is [`End::Shutdown`]'s path through [`wait_end`].

use std::borrow::Cow;
use std::fmt;
use std::io::{self, BufRead, BufReader, Read};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jinja::{ChatTemplate, TemplateError};
use serde::de::{Deserialize, Deserializer, Error, IgnoredAny, MapAccess, Visitor};
use serde_json::{Map, Value, json};

use crate::admit;
use crate::callstream::CallPiece;
use crate::constrain;
use crate::dsml::{ChatParser, MarkupError, Message, ToolCall, ToolFormat, Tools, parses_tools};
use crate::embed::Embed;
use crate::engine::{
    DraftProps, Engine, EngineError, EngineProps, ModelProps, PlacementProps, SamplerFactory,
    SamplingParams, StateError, Tokenizer,
};
use crate::event;
use crate::exit::Code;
use crate::flag::{ApiKeys, Denied};
use crate::genloop::{self, Depth, Event, GenError, GenParams, Outcome, Slot, Timings, ms_since};
use crate::glmxml::ArgTypes;
use crate::guard::{Guard, GuardError, Guarded};
use crate::http::{self, EventStream, Request};
use crate::media::{self, ExpandError, MediaError, Medium, Prompt, SharedMediaModel, VideoCtx};
use crate::models::Listed;
use crate::qwenxml::ParamKinds;
use crate::reasoning::{ReasoningFormat, ThinkEntry, ThinkTags};
use crate::sampling;
use crate::sched::{Board, Refusal, Reserve, SlotConfig, Use, default_depth};
use crate::slotfile;
use crate::swap::Park;
use crate::video::{self, VideoSupport};
use crate::worker::{self, Acted, Action, Events, Msg, ReqName, Round, Shared, Stats, Submit};

use self::listener::Listener;

/// Anthropic's Messages API, a child of this module so it runs the chat path's
/// own steps.
#[path = "anthropic.rs"]
mod anthropic;
/// The embeddings endpoints over the side model, a child of this module so
/// they read a prompt as the completion path does.
#[path = "embeddings.rs"]
pub mod embeddings;
/// The listening socket bound before the model loads, which answers `loading`
/// until a server takes it over; a child of this module so it serves through
/// the accept loop and the refusal written here.
#[path = "listener.rs"]
pub mod listener;
/// Per-token log-probabilities, which the generation loop collects and the
/// completion and chat answers render.
#[path = "logprobs.rs"]
pub(crate) mod logprobs;
/// OpenAI's text completion API and the chat token counts, a child of this
/// module so they run the completion and chat paths' own steps.
#[path = "oaicompl.rs"]
mod oaicompl;
/// The responses the Responses API keeps, a child of this module so it holds
/// a background response's client.
#[path = "response_store.rs"]
mod response_store;
/// OpenAI's Responses API, a child of this module so it runs the chat path's
/// own steps.
#[path = "responses.rs"]
mod responses;
/// The decision models' `POST /v1/systemone`, answered by the generative model
/// through the chat path's structured output, a child of this module so it runs
/// the chat path's own steps.
#[path = "systemone.rs"]
mod systemone;
#[cfg(test)]
#[path = "testserve.rs"]
mod testserve;

/// What the server says about itself and how it samples.
pub struct ServerConfig {
    /// `model` in responses and the `/v1/models` id.
    pub model_alias: String,
    /// `model_path` in `/props`.
    pub model_path: String,
    /// Jinja chat template source (GGUF `tokenizer.chat_template`).
    pub chat_template: String,
    /// The family's request preparation, which shapes a request before the
    /// template renders it; `None` renders the request as the template reads
    /// it.
    pub chat_preparation: Option<ChatPreparation>,
    /// Per-request sampler builder; `None` uses [`sampling::reference_factory`].
    pub sampler: Option<SamplerFactory>,
    /// How long `/health` keeps answering 503 after an engine error before
    /// [`Server::run`] returns.
    pub fatal_linger: Duration,
    /// The directory slot saves go to and restores read from
    /// (`--slot-save-path`); it must exist. `None` answers every slot action
    /// with 501.
    pub slot_save_path: Option<PathBuf>,
    /// The API keys the request check asks for (`--api-key`,
    /// `--api-key-file`); an empty set checks nothing.
    pub api_keys: ApiKeys,
}

/// A family's request preparation: it shapes a request before the template
/// renders it, and refuses it by name when it cannot, a 400 about the
/// request's `messages`.
///
/// The family supplies one with its chat template; `None` renders the request
/// as the template reads it.
pub type ChatPreparation = fn(&mut ChatRequest<'_>) -> Result<(), String>;

/// What a request preparation reads and writes: the messages as the template
/// will see them, the request body, the template's variables, and the guard
/// that marks client text.
pub struct ChatRequest<'a> {
    /// The messages, flattened, as the template will see them.
    pub messages: &'a mut Vec<Value>,
    /// The request body.
    pub body: &'a Map<String, Value>,
    /// The template's variables; the preparation adds the ones the template
    /// reads and the request body holds.
    pub vars: &'a mut Map<String, Value>,
    /// Marks a client text the way the request's other texts are marked.
    marker: &'a dyn Fn(&mut Value),
}

impl ChatRequest<'_> {
    /// Mark a client text: a special token's spelling in it stays text in the
    /// prompt, as in every other text the client sends.
    pub fn mark(&self, value: &mut Value) {
        (self.marker)(value);
    }
}

/// Connections served at once, each on a thread of its own. Past the requests
/// generating, these hold requests waiting for a slot, streams, keep-alive
/// clients and the cheap endpoints' pollers; a connection past them is
/// refused, never queued. An idle connection keeps its place for
/// [`KEEP_ALIVE_IDLE`] at most.
pub const MAX_CONNECTIONS: usize = 64;

/// What a refused connection's or request's 503 tells the client to wait, in
/// seconds: a place frees when any request ends.
const RETRY_AFTER_SECS: &str = "1";

/// How long a connection may send nothing of its next request (its first one
/// too) before it is closed: cpp-httplib's keep-alive timeout, which
/// llama-server keeps.
pub const KEEP_ALIVE_IDLE: Duration = Duration::from_secs(5);

/// How long one read inside a request may wait (llama-server's `--timeout`).
const REQUEST_READ: Duration = Duration::from_secs(600);

/// The longest a response write may make no progress.
const RESPONSE_WRITE: Duration = Duration::from_secs(10);

/// How long the accept loop waits after a failed `accept` that will fail again
/// at once: out of descriptors or memory, or the same failure twice in a row.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

/// `EMFILE` and `ENFILE`, the same numbers on Linux and the BSDs; std gives
/// them no `ErrorKind` of their own.
const EMFILE: i32 = 24;
const ENFILE: i32 = 23;
/// `EBADF`, `EFAULT` and `EINVAL` (also the same numbers): the listening socket
/// itself is wrong, so the next `accept` fails the same way.
const EBADF: i32 = 9;
const EFAULT: i32 = 14;
const EINVAL: i32 = 22;

/// How long a refused connection is written to, and how long a connection closed
/// after its answer is read from ([`linger_close`]).
const REFUSE_DRAIN: Duration = Duration::from_millis(200);

/// The default [`ServerConfig::fatal_linger`]: long enough for a client or a
/// supervisor to read the 503, short enough that the port frees promptly.
pub const FATAL_LINGER: Duration = Duration::from_secs(2);

/// An engine error that ended the server: the error and what the engine said
/// about itself when it happened. Its `Display` is the crash block.
#[derive(Debug, Clone)]
pub struct EngineFailure {
    /// [`Engine::describe`] at the failure.
    pub engine: String,
    /// The engine's message.
    pub error: String,
}

impl fmt::Display for EngineFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the engine failed; the server stops instead of serving it\n  engine: {}\n  error: {}",
            self.engine, self.error
        )
    }
}

/// Why the server could not start.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("listener: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Template(#[from] TemplateError),
    #[error("{0}")]
    Engine(EngineFailure),
    #[error("--slot-save-path {}: not a directory", .0.display())]
    SlotSavePath(PathBuf),
    /// The engine's vocabulary names no id that ends a generation.
    #[error("the engine's vocabulary names no stop id")]
    NoStops,
    /// A slot count or queue depth the server or the engine cannot serve.
    #[error("{0}")]
    Slots(String),
}

/// Why the accept loop ended.
pub(crate) enum End {
    Io(io::Error),
    Engine(EngineFailure),
    /// An orderly stop was asked of the server ([`Stop`]): [`wait_end`]
    /// performs it, the one owner of the stop.
    Shutdown(Stop),
}

/// What asked for the server's orderly stop: the one name its stderr line and
/// every refusal it causes carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The first `SIGINT`.
    Sigint,
    /// The first `SIGTERM`.
    Sigterm,
    /// A `POST /shutdown`, from this peer.
    Posted(SocketAddr),
}

impl fmt::Display for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stop::Sigint => write!(f, "SIGINT"),
            Stop::Sigterm => write!(f, "SIGTERM"),
            Stop::Posted(peer) => write!(f, "POST /shutdown from {peer}"),
        }
    }
}

/// A bound, not yet running server.
pub struct Server {
    listener: Listener,
    state: Arc<State>,
    /// The engine and its slots, until [`Server::run`] starts their thread.
    slot: Slot,
    /// The placement's word the `ready` event names, when a seat says one
    /// ([`Server::with_place`]).
    place: Option<String>,
}

/// The slot count `slots` asks of `engine`, or why it cannot be served.
fn check_slots(engine: &dyn Engine, slots: &SlotConfig) -> Result<usize, ServeError> {
    let n = slots.parallel;
    let refuse = |why: String| Err(ServeError::Slots(why));
    if n == 0 {
        return refuse("--parallel 0: the server needs one slot".to_owned());
    }
    if n > MAX_CONNECTIONS {
        return refuse(format!(
            "--parallel {n}: past the {MAX_CONNECTIONS} connections the server serves at once, \
             each running request holds one"
        ));
    }
    let declared = engine.slots();
    if n > declared {
        return refuse(format!(
            "--parallel {n}: this engine serves {declared} slot(s) at once"
        ));
    }
    if n > 1 && engine.turns() == Some(Park::Ids) && engine.media_model().is_some() {
        return refuse(format!(
            "--parallel {n}: this engine takes images and its slots take it in turns parked as \
             their ids, which carry no image to feed again"
        ));
    }
    let rows = engine.advance_rows();
    if n > 1 && rows > 1 && engine.turns().is_none() && !engine.slot_drafts() {
        return refuse(format!(
            "--parallel {n}: this engine drafts ({rows} rows a pass) and keeps no draft state \
             per slot (slot_drafts), so one slot's pass is not its own"
        ));
    }
    let most = default_depth(MAX_CONNECTIONS, n);
    let depth = slots.queue_depth.unwrap_or(most);
    if depth > most {
        return refuse(format!(
            "--queue-depth {depth}: past the {most} requests that can wait at once beside {n} \
             running under the {MAX_CONNECTIONS}-connection limit"
        ));
    }
    Ok(depth)
}

impl Server {
    /// Binds `addr` and takes ownership of the engine: one slot, the queue at
    /// its default depth.
    pub fn bind(
        addr: impl ToSocketAddrs,
        engine: Box<dyn Engine>,
        config: ServerConfig,
    ) -> Result<Self, ServeError> {
        Server::bind_with(addr, engine, config, SlotConfig::default())
    }

    /// Binds `addr` and takes ownership of the engine, serving the slots
    /// `slots` asks for. A slot count the engine does not declare
    /// ([`Engine::slots`]), several slots on an engine that drafts unless they
    /// take it in turns ([`Engine::turns`]) or it keeps its draft per slot
    /// ([`Engine::slot_drafts`]), several slots that take an engine of images
    /// ([`Engine::media_model`]) in turns parked as ids, and a depth no queue
    /// can reach are refused by name ([`ServeError::Slots`]), as is a chat
    /// template that does not parse or whose thinking probe does not render
    /// ([`ServeError::Template`]). The socket is bound after those checks and
    /// accepts nothing until the server runs.
    pub fn bind_with(
        addr: impl ToSocketAddrs,
        engine: Box<dyn Engine>,
        config: ServerConfig,
        slots: SlotConfig,
    ) -> Result<Self, ServeError> {
        Server::assemble(|| Listener::bind(addr), engine, config, slots)
    }

    /// [`bind_with`](Server::bind_with) on a listener bound earlier
    /// ([`Listener::early`]): it keeps answering `loading` until the server
    /// runs, then serves from the same socket. The refusals are
    /// `bind_with`'s, made before the listener is handed over.
    pub fn bind_listener(
        listener: Listener,
        engine: Box<dyn Engine>,
        config: ServerConfig,
        slots: SlotConfig,
    ) -> Result<Self, ServeError> {
        Server::assemble(|| Ok(listener), engine, config, slots)
    }

    /// The checks both binds make, then the server over the listener `open`
    /// yields at the point the socket used to be bound.
    fn assemble(
        open: impl FnOnce() -> io::Result<Listener>,
        engine: Box<dyn Engine>,
        config: ServerConfig,
        slots: SlotConfig,
    ) -> Result<Self, ServeError> {
        let template = ChatTemplate::parse(&config.chat_template)?;
        let thinking_off = probe_thinking_off(&template)?;
        if let Some(dir) = config.slot_save_path.as_ref().filter(|d| !d.is_dir()) {
            return Err(ServeError::SlotSavePath(dir.clone()));
        }
        let depth = check_slots(&*engine, &slots)?;
        let listener = open()?;
        let tok = engine.tokenizer();
        if tok.stops().is_empty() {
            return Err(ServeError::NoStops);
        }
        let engine_props = engine_object(&engine.props_engine());
        let media = engine.media_model();
        let (video, video_line) = VideoSupport::detect();
        if media.as_ref().is_some_and(|m| m.video_rule().is_some()) {
            eprintln!("bloomery-serve: {video_line}");
        }
        let info = ModelInfo {
            n_vocab: tok.n_vocab(),
            ctx_max: engine.ctx_max(),
            // A vocabulary with no BOS renders `bos_token` as empty text.
            bos_text: tok.bos().map(|id| tok.decode(&[id])).unwrap_or_default(),
            eos_text: tok.decode(&[tok.eos()]),
        };
        let parallel = slots.parallel;
        let shared = Shared::new(
            Board::new(parallel, depth, slots.picker),
            listener.end(),
            config.sampler.unwrap_or_else(sampling::reference_factory),
        );
        if let Some(note) = thinking_off.note() {
            eprintln!("bloomery-serve: {note}");
        }
        let vocab = Arc::new(constrain::Vocab::read(&*tok));
        let state = State {
            shared: Arc::new(shared),
            parallel,
            depth,
            tok,
            vocab,
            fatal_linger: config.fatal_linger,
            tool_format: ToolFormat::of_chat_template(&template),
            object_arguments: needs_object_arguments(&template),
            thinking_off,
            preparation: config.chat_preparation,
            think: ThinkTags::THINK,
            template,
            alias: config.model_alias,
            model_path: config.model_path,
            engine_props,
            media,
            images: media::PreparedImages::new(media::PREPARED_IMAGE_BYTES),
            video,
            slot_save_path: config.slot_save_path,
            side: None,
            api_keys: config.api_keys,
            info,
            start_unix: unix_now(),
            responses: response_store::Store::default(),
        };
        Ok(Server {
            listener,
            state: Arc::new(state),
            slot: Slot::with_slots(engine, parallel).schema_jump(slots.schema_jump),
            place: None,
        })
    }

    /// The bound address (the real port when bound to port 0).
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.listener.local_addr())
    }

    /// The server with `place` as the placement word its `ready` event names
    /// (`a`, `bp`, …), which only the seat that planned the placement knows.
    #[must_use]
    pub fn with_place(mut self, place: impl Into<String>) -> Server {
        self.place = Some(place.into());
        self
    }

    /// The server with `embed` beside its engine: the embeddings routes'
    /// model ([`crate::embed`]), listed second in `/v1/models`. Its facts are
    /// read here, once.
    #[must_use]
    pub fn with_side(mut self, embed: Box<dyn Embed>) -> Server {
        Arc::get_mut(&mut self.state)
            .expect("the state is shared only once the server starts")
            .side = Some(embeddings::Side::new(embed));
        self
    }

    /// The server with `video` as what decodes a clip, in place of the ffmpeg
    /// and ffprobe it found on `PATH` when it bound: a seat names the threads
    /// its engine's pool leaves ([`VideoSupport::threads`]), a test a decoder
    /// that makes frames from their indices.
    #[must_use]
    pub fn with_video(mut self, video: VideoSupport) -> Server {
        Arc::get_mut(&mut self.state)
            .expect("the state is shared only once the server starts")
            .video = video;
        self
    }

    /// The server with `think` as the tags of the model's think span, in place
    /// of `<think>` and `</think>`: a seat gives the pair its model's
    /// description names (`models::ReasoningFormat::ThinkSpan`). The reasoning
    /// split, the budget's start state and forced close, and the close a
    /// structured-output request writes first read them.
    #[must_use]
    pub fn with_think_tags(mut self, think: ThinkTags) -> Server {
        Arc::get_mut(&mut self.state)
            .expect("the state is shared only once the server starts")
            .think = think;
        self
    }

    /// Starts the engine thread and the accept loop, which serves connections,
    /// one thread each and at most [`MAX_CONNECTIONS`] at once, until the
    /// listener fails; returns the server's state and what ends it. The
    /// engine thread owns the engine from here and runs detached (its handle
    /// dropped): an orderly stop waits for the loop it leaves and then for
    /// every answer a connection is writing ([`Shared::wait_stop`]), never
    /// for the thread's end.
    fn start(self) -> io::Result<(Arc<State>, mpsc::Receiver<End>)> {
        let Server {
            listener,
            state,
            slot,
            place: _,
        } = self;
        let shared = Arc::clone(&state.shared);
        thread::Builder::new()
            .name("serve-engine".to_owned())
            .spawn(move || worker::serve(slot, shared))
            .map_err(|e| {
                io::Error::new(e.kind(), format!("cannot start the engine thread: {e}"))
            })?;
        let conn_state = Arc::clone(&state);
        let ended = listener.serve(move |stream| {
            keep_alive(
                stream,
                &conn_state.shared.answers,
                &conn_state.api_keys,
                |req, w| route(&conn_state, req, w),
            );
        });
        Ok((state, ended))
    }

    /// Serves until the listener fails, the engine does, or the server is
    /// asked to stop, and returns why. An orderly stop never returns: it ends
    /// the process inside [`exit_shutdown`], so every seat's
    /// `Ok(server.run())` is the error paths' alone. Once the listener
    /// serves, the `ready` event is printed ([`crate::event`]), the start
    /// screen is drawn when a seat armed it ([`crate::screen`]), and the live
    /// view takes its keys where it can ([`crate::live`]); the `stopped` event
    /// is printed before `run` returns or the process exits.
    pub fn run(mut self) -> ServeError {
        // The signals are installed before the listener serves: from the
        // first answer on, Ctrl+C or a kill stops the server the orderly way.
        install_signals({
            let shared = Arc::clone(&self.state.shared);
            move |stop| {
                if shared.begin_stop(&stop) {
                    let _ = shared.end.send(End::Shutdown(stop));
                }
            }
        });
        let addr = self.local_addr();
        let url = event::url(self.listener.local_addr());
        let place = self.place.take();
        let (state, ended) = match self.start() {
            Ok(s) => s,
            Err(e) => return ServeError::Io(e),
        };
        event::emit(&event::Event::Ready {
            url: &url,
            model: &state.alias,
            ctx: state.info.ctx_max as u64,
            slots: state.parallel as u64,
            place: place.as_deref(),
        });
        let live = match addr {
            Ok(addr) => crate::live::keys().then_some(addr),
            Err(e) => {
                eprintln!("bloomery-serve: live view: the address: {e}");
                None
            }
        };
        if !crate::screen::draw(live.is_some()) && live.is_some() {
            crate::screen::live_hint();
        }
        if let Some(addr) = live {
            start_live(&state, addr);
        }
        // `state` holds a sender, so the channel cannot close while we wait.
        match wait_end(&ended, state.fatal_linger) {
            Ended::Error(e) => {
                crate::live::restore();
                emit_stopped(&e);
                e
            }
            Ended::Stop(cause) => exit_shutdown(&cause, |bound| state.shared.wait_stop(bound)),
        }
    }

    /// Serves in the background and returns the address. What ends the
    /// server is not waited for: after an engine failure it answers 503 for
    /// as long as the process lives.
    pub fn spawn(self) -> io::Result<SocketAddr> {
        let addr = self.local_addr()?;
        self.start()?;
        Ok(addr)
    }
}

/// Starts the live view of the server `state` listening on `addr`
/// ([`crate::live`]); a failure is one line on stderr and the server serves on
/// without it.
fn start_live(state: &Arc<State>, addr: SocketAddr) {
    let routes = LISTED
        .into_iter()
        .filter(|&(_, p)| p != path::EMBEDDINGS || state.side.is_some())
        .collect();
    let now = {
        let state = Arc::clone(state);
        Box::new(move || live_now(&state))
    };
    let keyed = !state.api_keys.is_empty();
    if let Err(e) = crate::live::start(addr, state.alias.clone(), routes, keyed, now) {
        eprintln!("bloomery-serve: live view: {e}; the server serves on without it");
    }
}

/// A copy of what the live view shows of `state`, each lock held only to
/// copy.
fn live_now(state: &State) -> crate::live::Now {
    use crate::live::{Phase, Slot, Status};
    let (slots, queued) = {
        let b = relock(&state.shared.board);
        let slots = b
            .slots()
            .iter()
            .map(|s| {
                let phase = match s.state {
                    Use::Free => Phase::Idle,
                    Use::Held => Phase::Held,
                    Use::Running if s.view.n_decoded == 0 => Phase::Prefill,
                    Use::Running => Phase::Decode,
                };
                Slot {
                    phase,
                    n_past: s.view.n_past,
                    task: s.view.id_task,
                    n_decoded: s.view.n_decoded,
                }
            })
            .collect();
        (slots, b.waiting())
    };
    let (prompt_rate, decode_rate, cached, evaluated) = {
        let s = relock(&state.shared.stats);
        (
            s.prompt_rate(),
            s.decode_rate(),
            s.n_prompt_cached_total,
            s.n_prompt_total,
        )
    };
    let status = if relock(&state.shared.fatal).is_some() {
        Status::Failed
    } else if relock(&state.shared.stopping).is_some() {
        Status::Stopping
    } else {
        Status::Ready
    };
    crate::live::Now {
        status,
        slots,
        rates: Vec::new(),
        queued,
        room: state.room(),
        prompt_rate,
        decode_rate,
        cached,
        evaluated,
        requests: relock(&state.shared.done).newest_first(),
    }
}

struct ModelInfo {
    n_vocab: usize,
    ctx_max: usize,
    bos_text: String,
    eos_text: String,
}

struct State {
    /// The board, the counters and the engine's failure, which the engine
    /// thread shares.
    shared: Arc<Shared>,
    /// Slots.
    parallel: usize,
    /// The requests that may wait past the free slots: with the slots, the
    /// most choices one request may ask for ([`take_choices`]).
    depth: usize,
    /// The engine's vocabulary, read without the engine.
    tok: Arc<dyn Tokenizer>,
    /// What structured output needs of the vocabulary, read once when the
    /// server binds ([`crate::constrain::Vocab`]).
    vocab: Arc<constrain::Vocab>,
    fatal_linger: Duration,
    template: ChatTemplate,
    /// The tool-call markup the template teaches, read from its source once.
    tool_format: ToolFormat,
    /// Whether the template reads a tool call's `arguments` as an object
    /// ([`needs_object_arguments`]), decided once.
    object_arguments: bool,
    /// How the template answers `reasoning_effort: "none"`
    /// ([`probe_thinking_off`]), decided once.
    thinking_off: ThinkingOff,
    /// The family's request preparation, which shapes a request before the
    /// template renders it; `None` renders the request as the template reads
    /// it.
    preparation: Option<ChatPreparation>,
    /// The tags of the model's think span, `<think>` and `</think>` until a
    /// seat names others ([`Server::with_think_tags`]).
    think: ThinkTags,
    alias: String,
    model_path: String,
    /// `/props`' `engine` object, built when the server binds.
    engine_props: Value,
    /// The model's image input, `None` for an engine that takes none: an
    /// image part is rendered as the model's placeholder only when it exists,
    /// and refused by name otherwise. Read without the engine.
    media: Option<SharedMediaModel>,
    /// The images this server has already prepared for its model
    /// ([`media::PreparedImages`]): a request's already-sent images it still
    /// holds are served without a decode or a prepare and draw on none of the
    /// decode budget.
    images: media::PreparedImages,
    /// What decodes a clip: ffmpeg found at startup, a decoder a caller set
    /// ([`Server::with_video`]), or the reason there is none. A clip is taken
    /// when this and the model's video rule are both there ([`State::takes_video`]).
    video: VideoSupport,
    /// Where slot files live; `None` refuses slot actions.
    slot_save_path: Option<PathBuf>,
    /// The embedding model beside the engine ([`Server::with_side`]); `None`
    /// answers the embeddings routes with a 501.
    side: Option<embeddings::Side>,
    /// The API keys every request is checked against before it dispatches;
    /// an empty set checks nothing.
    api_keys: ApiKeys,
    info: ModelInfo,
    start_unix: u64,
    /// The responses `/v1/responses` keeps.
    responses: response_store::Store,
}

pub(crate) fn relock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A caught panic's message, as the failure it ends the server with names it (a model run beside
/// the engine names its own the same way).
pub fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_owned())
}

/// The arrival tickets of the board ([`crate::sched::Board`]): drawn in
/// arrival order, the turn passed on as each takes its slot.
///
/// `serving` is the oldest ticket whose turn has not come, `next` the next
/// ticket to hand out. `serving == next` is the idle state: every ticket
/// handed out has been served and nothing is queued. A poisoned lock is
/// recovered as everywhere here: a panic in one run must not wedge the queue.
#[derive(Default)]
pub(crate) struct SlotQueue {
    state: Mutex<QueueState>,
}

#[derive(Default)]
struct QueueState {
    /// The next ticket to hand out. Arrival order is the order tickets are
    /// drawn; a u64 cannot run out in a process's life.
    next: u64,
    /// The ticket whose turn it is.
    serving: u64,
}

impl SlotQueue {
    /// Draws the next ticket.
    pub(crate) fn draw(&self) -> u64 {
        let mut q = relock(&self.state);
        let ticket = q.next;
        q.next = ticket + 1;
        ticket
    }

    /// The ticket whose turn it is.
    pub(crate) fn serving(&self) -> u64 {
        relock(&self.state).serving
    }

    /// Passes the turn to the ticket after `ticket`.
    pub(crate) fn pass(&self, ticket: u64) {
        relock(&self.state).serving = ticket + 1;
    }

    /// Draws a ticket only when the queue is idle (`serving == next`: no
    /// ticket waits), so a caller that never waits cannot jump ahead of
    /// queued requests; `None` is that refusal.
    pub(crate) fn try_draw(&self) -> Option<u64> {
        let mut q = relock(&self.state);
        (q.serving == q.next).then(|| {
            let ticket = q.next;
            q.next = ticket + 1;
            ticket
        })
    }
}

impl State {
    /// The positions the slots can hold in all, every slot the whole
    /// `ctx_max`: the KV cache's room `/metrics`' usage ratio is read against.
    fn room(&self) -> usize {
        self.info.ctx_max * self.parallel
    }

    /// Whether a clip can be served: a decoder is there, and the chat model or
    /// the embedding model has a video rule ([`MediaModel::video_rule`]). The
    /// chat path also needs the model's video token, the embedding path its
    /// delimiters; a model without them refuses a clip by name.
    fn takes_video(&self) -> bool {
        let rule = |m: &SharedMediaModel| m.video_rule().is_some();
        self.video.is_ready()
            && (self.media.as_ref().is_some_and(rule)
                || self.side.as_ref().is_some_and(|s| s.takes_video()))
    }

    fn next_id(&self) -> u64 {
        self.shared.next_id()
    }

    /// A 32-hex-digit id, unique per process.
    fn random_id(&self) -> String {
        let n = self.next_id();
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        format!("{:016x}{:016x}", mix(t ^ n), id_half(process_seed(), n))
    }

    fn fatal(&self) -> Option<String> {
        relock(&self.shared.fatal).clone()
    }

    /// The orderly stop's cause, once it began ([`Shared::begin_stop`]).
    fn stopping(&self) -> Option<String> {
        relock(&self.shared.stopping).clone()
    }
}

/// The id's second half: the request counter under the process's seed, never
/// the counter alone — with the counter alone every process drew the same
/// half at the same request number.
fn id_half(seed: u64, n: u64) -> u64 {
    mix(seed ^ n.wrapping_add(0x5bd1_e995))
}

/// A seed no other process shares, drawn once. The workspace links no OS
/// entropy source, so it mixes what the OS varies per process: the clock, the
/// pid, and this static's own address (its placement varies with the
/// process's address space).
fn process_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let pid = u64::from(std::process::id());
        let addr = std::ptr::addr_of!(SEED) as u64;
        mix(mix(t) ^ pid) ^ mix(addr)
    })
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------- connection

/// What the accept loop does after `accept` failed with `e` for the
/// `repeats + 1`-th time in a row: `None` ends the server (the listener no
/// longer stands, or the error is the listening socket's own), else the wait
/// before the next `accept`. A connection that failed before it was taken
/// (aborted, reset, a network error it carried) is that client's; the loop goes
/// on at once, and waits only when the failure repeats or cannot clear by
/// itself (out of descriptors or memory).
fn after_accept_error(e: &io::Error, listener_stands: bool, repeats: u32) -> Option<Duration> {
    if !listener_stands || matches!(e.raw_os_error(), Some(EBADF | EFAULT | EINVAL)) {
        return None;
    }
    let exhausted =
        matches!(e.raw_os_error(), Some(EMFILE | ENFILE)) || e.kind() == io::ErrorKind::OutOfMemory;
    Some(if exhausted || repeats > 0 {
        ACCEPT_BACKOFF
    } else {
        Duration::ZERO
    })
}

/// One live connection's place under its limit, given back when its
/// thread ends, by a return or a panic.
struct Permit(Arc<AtomicUsize>);

impl Permit {
    /// A place, or `None` when `limit` connections are live.
    fn take(live: &Arc<AtomicUsize>, limit: usize) -> Option<Permit> {
        let held = live.fetch_add(1, Ordering::SeqCst);
        let permit = Permit(Arc::clone(live));
        (held < limit).then_some(permit)
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Accepts connections on `listener` on a thread of its own and serves each
/// through `serve` ([`admit`]). An `accept` that fails while the listener
/// stands is a named line on stderr and the loop goes on
/// ([`after_accept_error`]); a listener that no longer stands is sent on `end`
/// and ends the loop. Every server of the crate accepts here.
pub(crate) fn accept<F>(listener: TcpListener, end: mpsc::Sender<End>, serve: F)
where
    F: Fn(TcpStream) + Send + Sync + 'static,
{
    let port = listener.local_addr().map_or(0, |a| a.port());
    let live = Arc::new(AtomicUsize::new(0));
    let refusing = Arc::new(AtomicUsize::new(0));
    let serve = Arc::new(serve);
    thread::spawn(move || {
        let mut repeats = 0;
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    repeats = 0;
                    admit(&serve, &live, &refusing, port, stream);
                }
                Err(e) => {
                    let stands = listener.local_addr().is_ok();
                    let Some(wait) = after_accept_error(&e, stands, repeats) else {
                        let gone = io::Error::new(
                            e.kind(),
                            format!(
                                "accept failed ({e}); the listener cannot accept again \
                                 (it stands: {stands})"
                            ),
                        );
                        let _ = end.send(End::Io(gone));
                        return;
                    };
                    eprintln!(
                        "bloomery-serve: accept failed ({e}); the listener stands, next accept in {} ms",
                        wait.as_millis()
                    );
                    repeats = repeats.saturating_add(1);
                    thread::sleep(wait);
                }
            }
        }
    });
}

/// What `wait_end` heard: the error `run` returns, or the orderly stop it
/// performs instead of returning.
pub(crate) enum Ended {
    Error(ServeError),
    Stop(Stop),
}

/// What ends a server, waited for on `ended`: the listener's failure at once,
/// an engine's after `linger`, during which `/health` answers it
/// ([`fatal_health`]), an orderly stop at once ([`exit_shutdown`]). The caller
/// holds a sender, so the channel cannot close while it waits.
pub(crate) fn wait_end(ended: &mpsc::Receiver<End>, linger: Duration) -> Ended {
    match ended.recv() {
        Ok(End::Io(e)) => Ended::Error(ServeError::Io(e)),
        Ok(End::Engine(f)) => {
            thread::sleep(linger);
            Ended::Error(ServeError::Engine(f))
        }
        Ok(End::Shutdown(stop)) => Ended::Stop(stop),
        Err(mpsc::RecvError) => {
            Ended::Error(ServeError::Io(io::Error::other("accept loop vanished")))
        }
    }
}

/// Prints the `stopped` event of a run that ended in `e`, before the run
/// returns it: the engine's failure or the listener's, with the status the
/// process then exits with ([`Code::of`]). Both servers call it.
pub(crate) fn emit_stopped(e: &ServeError) {
    let cause = if matches!(e, ServeError::Engine(_)) {
        event::Cause::Engine
    } else {
        event::Cause::Listener
    };
    event::emit(&event::Event::Stopped {
        cause,
        exit: Code::of(e),
    });
}

// ---------------------------------------------------------------- orderly stop

/// How long an orderly stop waits, in all, before the process exits: on the
/// generative server for the engine thread to leave its loop and then for
/// every answer in flight to be written ([`Shared::wait_stop`]), on the
/// decision server for its answers in flight ([`Answers::wait`]). The worker
/// reads the stop before each action, start, planned prompt call and decode
/// round of its rounds (when its slots take turns, at each round's top and
/// before the running request's step), so the wait covers the one it is in
/// when the stop arrives, then the halted requests' booking and their last
/// messages written. A decode step sits far under the bound; a prompt run as
/// one engine call has none of its own, so a long one can outlast it, and
/// the process then exits with the engine thread still inside the call, its
/// stop line saying so, as it says when answers were still being written.
/// The loop's end comes after the engine is closed ([`crate::Engine::close`]),
/// and nothing after it is waited for: the engine thread leaves the engine to the
/// process exit ([`crate::worker::serve`]).
const ENGINE_STOP: Duration = Duration::from_secs(5);

/// The signal handlers' whole state, the only things a handler touches: the
/// pipe end a first signal writes one byte to, and whether a first signal was
/// already taken — both async-signal-safe to reach (one atomic, one `write`).
struct Signals {
    /// The pipe's write end.
    w: libc::c_int,
    taken: AtomicBool,
}

static SIGNALS: OnceLock<Signals> = OnceLock::new();

/// Whether the signal handlers were installed: once a process. The first
/// server to `run` owns the signals; a later one's `run` leaves them as they
/// are, so its own stop is not asked for by them.
static SIGNALS_ARMED: AtomicBool = AtomicBool::new(false);

/// Installs the `SIGINT`/`SIGTERM` handlers, llama-server's shape: the first
/// signal becomes the server's orderly stop through `on_first`; a second one
/// terminates the process at once, in case the stop hangs. The handler itself
/// does only async-signal-safe work — one atomic and one `write` to the pipe
/// this function owns; everything else (`begin_stop`, the channel send)
/// happens on the reader thread spawned here, which `on_first` runs on.
pub(crate) fn install_signals(on_first: impl Fn(Stop) + Send + 'static) {
    if SIGNALS_ARMED.swap(true, Ordering::SeqCst) {
        return;
    }
    let mut fds = [0 as libc::c_int; 2];
    // SAFETY: `fds` is a valid two-int array the call may write to.
    let piped = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(piped, 0, "cannot make the signal handlers' pipe: {piped}");
    let _ = SIGNALS.set(Signals {
        w: fds[1],
        taken: AtomicBool::new(false),
    });
    let reader = thread::Builder::new()
        .name("serve-signals".to_owned())
        .spawn(move || {
            let mut byte = [0u8; 1];
            loop {
                // SAFETY: `byte` is a valid one-byte buffer the call may write
                // to; the fd is this pair's read end, owned until the thread
                // ends.
                let read = unsafe { libc::read(fds[0], byte.as_mut_ptr().cast(), 1) };
                match read {
                    1 => match i32::from(byte[0]) {
                        libc::SIGINT => on_first(Stop::Sigint),
                        libc::SIGTERM => on_first(Stop::Sigterm),
                        _ => {}
                    },
                    // The pipe cannot fill (one byte a process), so a failed
                    // read is EINTR or a broken pipe: the former reads again,
                    // the latter leaves the signals to the default.
                    -1 if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted => {}
                    _ => return,
                }
            }
        });
    if let Err(e) = reader {
        panic!("cannot start the signal reader thread: {e}");
    }
    handle(libc::SIGINT, on_signal as *const () as usize);
    handle(libc::SIGTERM, on_signal as *const () as usize);
}

/// Registers `action` (a handler, `SIG_DFL` or `SIG_IGN`) for `sig` with
/// `SA_RESTART`, so a signal arriving mid-`accept` or mid-`read` restarts the
/// call instead of failing it: the stop is carried by the pipe, not by an
/// EINTR. Async-signal-safe (`sigaction` only).
pub(crate) fn handle(sig: libc::c_int, action: libc::sighandler_t) {
    // SAFETY: an all-zero `sigaction` is a valid value — a null handler, an
    // empty mask, no flags — and only the fields set below are changed.
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    act.sa_sigaction = action;
    act.sa_flags = libc::SA_RESTART;
    // SAFETY: `act` is fully initialized and a valid `sigaction` for the
    // call; the old action is not asked for.
    unsafe { libc::sigaction(sig, &act, std::ptr::null_mut()) };
}

/// The `SIGINT`/`SIGTERM` handler: the first signal writes its number on the
/// pipe for the reader thread; a second terminates at once, as llama-server's
/// does. Async-signal-safe only: two atomics and two `write`s.
extern "C" fn on_signal(sig: libc::c_int) {
    let Some(s) = SIGNALS.get() else {
        return;
    };
    if s.taken.swap(true, Ordering::SeqCst) {
        crate::live::restore_from_signal();
        const LINE: &[u8] =
            b"bloomery-serve: Received second interrupt, terminating immediately.\n";
        // SAFETY: `write` and `_exit` are async-signal-safe; the line is a
        // static buffer, not touched again.
        unsafe {
            libc::write(libc::STDERR_FILENO, LINE.as_ptr().cast(), LINE.len());
            libc::_exit(1);
        }
    }
    let byte = sig as u8;
    // SAFETY: `byte` is a valid one-byte buffer for the write; the fd is the
    // pipe's write end, which lives until the process ends. A failed write
    // (the reader thread gone) leaves this signal uncarried — the stop then
    // still has its other path, `POST /shutdown`.
    unsafe { libc::write(s.w, &byte as *const u8 as *const libc::c_void, 1) };
}

/// How the orderly stop's wait ended ([`Shared::wait_stop`],
/// [`Answers::wait`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Waited {
    /// Every answer in flight was written, the engine thread's loop left
    /// first where there is one.
    Done,
    /// The bound ran out with the engine thread still in its loop, inside
    /// the work the stop arrived in, the requests it holds not yet answered.
    Loop,
    /// This many answers were still being written when the bound ran out.
    Answers(usize),
}

/// The orderly stop's last act, the one owner of the process's exit: the
/// server's `wait` inside [`ENGINE_STOP`] — the generative server's engine
/// loop and answers ([`Shared::wait_stop`]), the decision server's answers
/// ([`Answers::wait`]) — then one stderr line ([`stop_line`]) and success.
/// Called from `run`, after the stop's initiator already refused new work
/// and told the engine thread to end, so no seat's `Ok(server.run())` return
/// is reached by it. The exit tears the process down (its exit handlers and
/// the loaded libraries' destructors, the device driver's among them) only
/// after a wait that ended: past its loop the engine thread calls the driver
/// no more ([`crate::worker::serve`]), and no answer is left running. After a
/// wait that ran out, the engine thread or an answer can still be inside a
/// driver call, and the driver torn down beside it can corrupt the heap: the
/// process then ends at once, stdout flushed first.
pub(crate) fn exit_shutdown(cause: &Stop, wait: impl FnOnce(Duration) -> Waited) -> ! {
    crate::live::restore();
    let waited = wait(ENGINE_STOP);
    let ended = waited == Waited::Done;
    eprintln!("{}", stop_line(cause, waited));
    event::emit(&event::Event::Stopped {
        cause: match cause {
            Stop::Sigint | Stop::Sigterm => event::Cause::Signal,
            Stop::Posted(_) => event::Cause::Shutdown,
        },
        exit: Code::Ok,
    });
    if !ended {
        if let Err(e) = io::Write::flush(&mut io::stdout()) {
            eprintln!("bloomery-serve: stdout: {e}");
        }
        // SAFETY: `_exit` takes no pointer and does not return; it ends the
        // process without its exit handlers and destructors.
        unsafe { libc::_exit(0) }
    }
    std::process::exit(0)
}

/// The stop's stderr line: its cause, and what the wait left undone when
/// [`ENGINE_STOP`] ran out first.
pub(crate) fn stop_line(cause: &Stop, waited: Waited) -> String {
    let s = ENGINE_STOP.as_secs();
    let tail = match waited {
        Waited::Loop => format!("; the engine thread did not end within {s} s"),
        Waited::Answers(n) => {
            format!("; {n} answer(s) were still being written at the {s} s bound")
        }
        Waited::Done => String::new(),
    };
    format!("bloomery-serve: shutdown ({cause}){tail}")
}

/// The message of the 503 every request that needs the engine gets once the
/// server's orderly stop began, whichever server of the crate it asked.
pub(crate) fn stopping_message(cause: &str) -> String {
    format!("the server is shutting down ({cause}); no new request is admitted")
}

/// `/health`'s 503 body once an engine failure ends the server.
pub(crate) fn fatal_health(reason: &str) -> Value {
    let mut v = error_body(503, "unavailable_error", reason);
    v["status"] = json!("error");
    v["reason"] = json!(reason);
    v
}

/// The message of the 503 every request that needs the engine gets once an
/// engine failure ends the server.
pub(crate) fn stopping(reason: &str) -> String {
    format!("the engine failed and the server is stopping: {reason}")
}

/// Serves `stream` on a thread named `serve:<port>`, or writes a 503 on the
/// listener's one refusal thread when [`MAX_CONNECTIONS`] are live. A busy
/// refusal thread or a failed spawn closes the connection at once.
fn admit<F>(
    serve: &Arc<F>,
    live: &Arc<AtomicUsize>,
    refusing: &Arc<AtomicUsize>,
    port: u16,
    stream: TcpStream,
) where
    F: Fn(TcpStream) + Send + Sync + 'static,
{
    let (permit, full) = match Permit::take(live, MAX_CONNECTIONS) {
        Some(permit) => (permit, false),
        None => {
            let Some(permit) = Permit::take(refusing, 1) else {
                return;
            };
            (permit, true)
        }
    };
    let serve = Arc::clone(serve);
    let spawned = thread::Builder::new()
        .name(if full {
            format!("refuse:{port}")
        } else {
            format!("serve:{port}")
        })
        .spawn(move || {
            let _permit = permit;
            if full {
                refuse(
                    stream,
                    &format!("the server is serving {MAX_CONNECTIONS} connections, its limit"),
                );
            } else {
                serve(stream);
            }
        });
    if let Err(e) = spawned {
        eprintln!("bloomery-serve: cannot start a connection thread: {e}");
    }
}

/// Answers a connection the server does not serve with a 503 carrying
/// `message`, as [`refuse_with`] does.
fn refuse(stream: TcpStream, message: &str) {
    refuse_with(stream, &error_body(503, "unavailable_error", message));
}

/// Answers a connection the server does not serve with a 503 carrying `body`
/// and `Retry-After`, then closes it through [`linger_close`]. Runs on the
/// connection's thread, never the listener's accept loop.
fn refuse_with(mut stream: TcpStream, body: &Value) {
    if stream.set_write_timeout(Some(REFUSE_DRAIN)).is_err() {
        return;
    }
    let sent = http::respond(
        &mut stream,
        &closing(),
        503,
        JSON,
        &[("Retry-After", RETRY_AFTER_SECS.to_owned())],
        body.to_string().as_bytes(),
    );
    if sent.is_ok() {
        linger_close(&stream);
    }
}

/// Ends a connection whose answer is written while request bytes may still be
/// arriving: shuts the write side, then reads and drops what the client sends
/// for at most about [`REFUSE_DRAIN`] twice and [`http::MAX_BODY`] bytes, so the
/// close does not reset the connection before the client reads the answer
/// (RFC 9112 9.6). The caller drops the stream afterwards.
fn linger_close(stream: &TcpStream) {
    if stream.shutdown(Shutdown::Write).is_err()
        || stream.set_read_timeout(Some(REFUSE_DRAIN)).is_err()
    {
        return;
    }
    let until = Instant::now() + REFUSE_DRAIN;
    let mut sink = [0u8; 4096];
    let mut left = http::MAX_BODY;
    let mut r = stream;
    while left > 0 && Instant::now() < until {
        match r.read(&mut sink) {
            Ok(0) | Err(_) => return,
            Ok(n) => left = left.saturating_sub(n),
        }
    }
}

/// The request an answer stands on when none was read: HTTP/1.1, closing.
pub(crate) fn closing() -> Request {
    Request {
        method: String::new(),
        path: String::new(),
        query: Vec::new(),
        http10: false,
        headers: Vec::new(),
        body: Vec::new(),
        keep_alive: false,
        lease: Mutex::default(),
    }
}

/// Waits at most [`KEEP_ALIVE_IDLE`] for the first byte of the connection's
/// next request (a byte already buffered counts), then gives the rest of the
/// request [`REQUEST_READ`] per read. `false` when nothing came: the peer
/// closed, the wait ran out or the read failed, and the connection closes.
fn next_request_arrives(r: &mut BufReader<TcpStream>) -> bool {
    if !r.buffer().is_empty() {
        return true;
    }
    if r.get_ref().set_read_timeout(Some(KEEP_ALIVE_IDLE)).is_err() {
        return false;
    }
    let arrived = r.fill_buf().is_ok_and(|b| !b.is_empty());
    arrived && r.get_ref().set_read_timeout(Some(REQUEST_READ)).is_ok()
}

/// The answers a server's connections are writing: each request
/// [`keep_alive`] dispatches counts from its dispatch until its answer is
/// written, its client gone or its thread panicked. What every server's
/// orderly stop waits for before the process exits ([`exit_shutdown`]).
#[derive(Default)]
pub(crate) struct Answers {
    n: Mutex<usize>,
    /// Wakes [`Answers::wait`] when the last answer in flight is written.
    written: Condvar,
}

impl Answers {
    /// One answer in flight until the guard drops.
    fn answering(&self) -> Answering<'_> {
        *relock(&self.n) += 1;
        Answering(self)
    }

    /// Waits until no answer is in flight, at most until `until`.
    pub(crate) fn wait(&self, until: Instant) -> Waited {
        let n = relock(&self.n);
        let left = until.saturating_duration_since(Instant::now());
        let (n, _) = self
            .written
            .wait_timeout_while(n, left, |n| *n > 0)
            .unwrap_or_else(|e| e.into_inner());
        match *n {
            0 => Waited::Done,
            n => Waited::Answers(n),
        }
    }

    /// The answers in flight now.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        *relock(&self.n)
    }
}

/// An answer in flight ([`Answers`]); its drop counts it written.
struct Answering<'a>(&'a Answers);

impl Drop for Answering<'_> {
    fn drop(&mut self) {
        let mut n = relock(&self.0.n);
        *n -= 1;
        if *n == 0 {
            self.0.written.notify_all();
        }
    }
}

/// Answers `stream`'s requests in order through `route` while the client keeps
/// the connection alive ([`next_request_arrives`]); `route` answers one
/// request, `Ok(true)` when the connection may carry another, counted in
/// `answers` until it returns. Headers are authorized against `keys` before
/// continuing or reading the body; a denied or unreadable request closes the connection,
/// as does an answer that left the request's body unread, each through [`linger_close`].
pub(crate) fn keep_alive(
    stream: TcpStream,
    answers: &Answers,
    keys: &ApiKeys,
    mut route: impl FnMut(&Request, &mut TcpStream) -> io::Result<bool>,
) {
    let _ = stream.set_nodelay(true);
    if stream.set_write_timeout(Some(RESPONSE_WRITE)).is_err() {
        return;
    }
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::new(stream);
    loop {
        if !next_request_arrives(&mut r) {
            return;
        }
        let (req, unread, meter) = match (|| -> io::Result<Option<(Request, bool, _)>> {
            let Some(mut req) = http::read_head(&mut r)? else {
                return Ok(None);
            };
            let meter = match keys.check(
                &req.method,
                &req.path,
                req.header("Authorization"),
                req.header("X-Api-Key"),
            ) {
                Ok(meter) => meter,
                Err(denied) => {
                    refuse_key(denied, &mut req, &mut w)?;
                    linger_close(r.get_ref());
                    return Ok(None);
                }
            };
            let mut unread = false;
            if req.method == "OPTIONS" || matches!(req.path.as_str(), "/health" | "/v1/health") {
                // These public handlers ignore bodies; unread framing cannot carry a next request.
                unread = req.header("transfer-encoding").is_some()
                    || http::content_length(&req)?.is_some_and(|len| len > 0);
                req.keep_alive &= !unread;
            } else {
                http::read_body(&mut r, &mut w, &mut req)?;
            }
            Ok(Some((req, unread, meter)))
        })() {
            Ok(Some(read)) => read,
            Ok(None) => return,
            Err(e) => {
                let (status, headers) = match e.kind() {
                    io::ErrorKind::InvalidData => (400, Vec::new()),
                    // A body's content coding the server does not read.
                    io::ErrorKind::Unsupported => (
                        415,
                        vec![("Accept-Encoding", http::BODY_CODINGS.to_owned())],
                    ),
                    _ => return,
                };
                let body = error_body(status, "invalid_request_error", &e.to_string());
                let sent = http::respond(
                    &mut w,
                    &closing(),
                    status,
                    JSON,
                    &headers,
                    body.to_string().as_bytes(),
                );
                if sent.is_ok() {
                    linger_close(r.get_ref());
                }
                return;
            }
        };
        let answered = {
            let _answering = answers.answering();
            // A decide server's seated route is a decision, whatever its path.
            let takes_place = keys.inference_routes.contains(&req.path.as_str())
                || !responses::store_call(&req.method, &req.path);
            match meter
                .map(|m| m.admit(Instant::now(), takes_place))
                .transpose()
            {
                Ok(lease) => {
                    *relock(&req.lease) = lease.flatten();
                    route(&req, &mut w)
                }
                Err(refusal) => send_refusal(&mut w, &req, &refusal),
            }
        };
        match answered {
            Ok(true) if req.keep_alive => {}
            Ok(_) if unread => return linger_close(r.get_ref()),
            _ => return,
        }
    }
}

pub(crate) const JSON: &str = "application/json; charset=utf-8";

/// An error that becomes an OpenAI-style error object.
struct ApiError {
    code: u16,
    kind: &'static str,
    message: String,
    /// Sent with `Retry-After`: the request may succeed later as it is.
    retry_after: bool,
    /// The top-level request field the error is about, sent as `param`.
    param: Option<String>,
}

impl ApiError {
    /// This error as one about the request field `path` names (`top_k`,
    /// `messages[0].content`, `text.format`): its `param` is the path's
    /// top-level key.
    fn at(mut self, path: &str) -> Self {
        let end = path.find(['.', '[']).unwrap_or(path.len());
        self.param = Some(path[..end].to_owned());
        self
    }

    /// The error object: [`error_body`], with `param` when the error is about
    /// a request field.
    fn body(&self) -> Value {
        let mut v = error_body(self.code, self.kind, &self.message);
        if let Some(p) = &self.param {
            v["error"]["param"] = json!(p);
        }
        v
    }
}

impl From<::embed::request::Error> for ApiError {
    fn from(error: ::embed::request::Error) -> Self {
        use ::embed::request::Error;
        let e = ApiError {
            retry_after: false,
            param: None,
            code: 400,
            kind: match &error {
                Error::Context { .. } => "exceed_context_size_error",
                Error::Invalid(_) | Error::Field { .. } => "invalid_request_error",
            },
            message: error.to_string(),
        };
        match &error {
            Error::Field { field, .. } => e.at(field),
            _ => e,
        }
    }
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError {
        retry_after: false,
        param: None,
        code: 400,
        kind: "invalid_request_error",
        message: message.into(),
    }
}

/// The 404 of a route, model or stored response the server does not have.
fn not_found(message: impl Into<String>) -> ApiError {
    ApiError {
        retry_after: false,
        param: None,
        code: 404,
        kind: "not_found_error",
        message: message.into(),
    }
}

pub(crate) fn error_body(code: u16, kind: &str, message: &str) -> Value {
    json!({ "error": { "code": code, "message": message, "type": kind } })
}

fn send_json(w: &mut TcpStream, req: &Request, status: u16, v: &Value) -> io::Result<bool> {
    http::respond(w, req, status, JSON, &[], v.to_string().as_bytes())?;
    Ok(true)
}

fn send_error(w: &mut TcpStream, req: &Request, e: &ApiError) -> io::Result<bool> {
    let body = e.body().to_string();
    let retry = [("Retry-After", RETRY_AFTER_SECS.to_owned())];
    let headers: &[(&str, String)] = if e.retry_after { &retry } else { &[] };
    http::respond(w, req, e.code, JSON, headers, body.as_bytes())?;
    Ok(true)
}

/// The headers of the 204 an `OPTIONS` request gets: any origin may send
/// `GET`, `POST`, `DELETE` and `OPTIONS` with any header.
pub(crate) fn cors_preflight() -> [(&'static str, String); 2] {
    [
        (
            "Access-Control-Allow-Methods",
            "GET, POST, DELETE, OPTIONS".to_owned(),
        ),
        ("Access-Control-Allow-Headers", "*".to_owned()),
    ]
}

/// The one stderr line a request the key check refuses writes: its method,
/// path, peer and why, and never the key it sent.
fn deny_line(method: &str, path: &str, peer: &str, denied: Denied) -> String {
    let why = match denied {
        Denied::Key => "no valid API key",
        Denied::Scope => "the API key's scope does not reach the route",
    };
    format!("{method} {path} from {peer}: refused, {why}")
}

/// Answers a denied request's headers and marks the connection closed: an
/// unknown key with llama-server's 401, a key whose scope does not reach the
/// route with a 403. The unread body is never drained or interpreted as
/// another request.
fn refuse_key(denied: Denied, req: &mut Request, w: &mut TcpStream) -> io::Result<()> {
    let peer = w
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|e| format!("a peer that cannot be read: {e}"));
    eprintln!("{}", deny_line(&req.method, &req.path, &peer, denied));
    req.keep_alive = false;
    let (code, kind, message) = match denied {
        Denied::Key => (401, "authentication_error", "Invalid API Key".to_owned()),
        Denied::Scope => (
            403,
            "permission_error",
            format!(
                "this API key's scope does not reach {} {}",
                req.method, req.path
            ),
        ),
    };
    send_error(
        w,
        req,
        &ApiError {
            retry_after: false,
            param: None,
            code,
            kind,
            message,
        },
    )?;
    Ok(())
}

/// Answers a request its key's limits refused ([`crate::admit`]) with a 429
/// whose `Retry-After` is the refusal's wait; the body was read, so the
/// connection may carry another request.
fn send_refusal(w: &mut TcpStream, req: &Request, refusal: &admit::Refusal) -> io::Result<bool> {
    let body = error_body(429, "rate_limit_error", &refusal.message()).to_string();
    let retry = [("Retry-After", refusal.retry_after().to_string())];
    http::respond(w, req, 429, JSON, &retry, body.as_bytes())?;
    Ok(true)
}

/// The paths of the routes the live view's Connect tab lists
/// ([`crate::live`]): each is the first path its arm in [`route`] matches.
pub(crate) mod path {
    pub(crate) const HEALTH: &str = "/health";
    pub(crate) const MODELS: &str = "/v1/models";
    pub(crate) const PROPS: &str = "/props";
    pub(crate) const SLOTS: &str = "/slots";
    pub(crate) const METRICS: &str = "/metrics";
    pub(crate) const COMPLETION: &str = "/completion";
    pub(crate) const COMPLETIONS: &str = "/v1/completions";
    pub(crate) const CHAT: &str = "/v1/chat/completions";
    pub(crate) const MESSAGES: &str = "/v1/messages";
    pub(crate) const RESPONSES: &str = "/v1/responses";
    pub(crate) const EMBEDDINGS: &str = "/v1/embeddings";
    pub(crate) const TOKENIZE: &str = "/tokenize";
    pub(crate) const DETOKENIZE: &str = "/detokenize";
}

/// The routes the live view lists, by method and [`path`].
pub(crate) const LISTED: [(&str, &str); 13] = [
    ("GET", path::HEALTH),
    ("GET", path::MODELS),
    ("GET", path::PROPS),
    ("GET", path::SLOTS),
    ("GET", path::METRICS),
    ("POST", path::COMPLETION),
    ("POST", path::COMPLETIONS),
    ("POST", path::CHAT),
    ("POST", path::MESSAGES),
    ("POST", path::RESPONSES),
    ("POST", path::EMBEDDINGS),
    ("POST", path::TOKENIZE),
    ("POST", path::DETOKENIZE),
];

/// Dispatches one request; `Ok(true)` when the connection may carry another.
/// A new model route or alias also belongs in `Scope::of`'s `MODEL_ROUTES`
/// and the route list in `a_key_scope_reaches_its_routes_and_no_other`.
fn route(state: &Arc<State>, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let r = match (req.method.as_str(), req.path.as_str()) {
        ("GET", path::HEALTH | "/v1/health") => return health(state, req, w),
        ("GET", path::MODELS | "/models" | "/api/tags") => Ok(models(state)),
        ("GET", p) if let Some(id) = crate::models::named(p) => id
            .map_err(invalid)
            .and_then(|id| crate::models::lookup(&models(state), &id).map_err(not_found)),
        ("POST", "/api/show") => body(req).and_then(|b| show(state, &b)),
        ("GET", path::PROPS) => Ok(props(state)),
        ("GET", path::SLOTS) => Ok(slots(state)),
        ("GET", path::METRICS) => return metrics(state, req, w),
        ("POST", p) if p.starts_with("/slots/") => {
            return slot_action(state, req, w, &p["/slots/".len()..]);
        }
        ("POST", "/residency/reset") => return residency_reset(state, req, w),
        ("POST", "/shutdown") => return post_shutdown(state, req, w),
        ("POST", path::COMPLETION | "/completions") => return completion(state, req, w),
        ("POST", path::COMPLETIONS) => return oaicompl::completions(state, req, w),
        ("POST", path::CHAT | "/chat/completions") => return chat(state, req, w),
        ("POST", "/v1/chat/completions/input_tokens" | "/chat/completions/input_tokens") => {
            return oaicompl::count_tokens(state, req, w);
        }
        ("POST", p) if systemone::ROUTES.contains(&p) => return systemone::post(state, req, w),
        ("POST", path::MESSAGES) => return anthropic::messages(state, req, w),
        ("POST", "/v1/messages/count_tokens") => return anthropic::count_tokens(state, req, w),
        ("POST", path::RESPONSES | "/responses") => return responses::create(state, req, w),
        ("POST", "/v1/responses/input_tokens" | "/responses/input_tokens") => {
            return responses::input_tokens(state, req, w);
        }
        ("GET" | "DELETE" | "POST", p) if responses::stored_id(p).is_some() => {
            return responses::stored(state, req, w);
        }
        ("POST", path::EMBEDDINGS) => return embeddings::post(state, req, w, true),
        ("POST", "/embeddings" | "/embedding") => return embeddings::post(state, req, w, false),
        ("POST", path::TOKENIZE) => body(req).and_then(|b| tokenize(state, &b)),
        ("POST", path::DETOKENIZE) => body(req).and_then(|b| detokenize(state, &b)),
        ("POST", "/apply-template") => body(req).and_then(|b| apply_template(state, &b)),
        ("OPTIONS", _) => {
            http::respond(w, req, 204, "text/plain", &cors_preflight(), b"")?;
            return Ok(true);
        }
        _ => Err(not_found(format!(
            "{}; POST {} answers a SystemOne decision request",
            req.not_found(),
            systemone::ROUTES.join(", POST ")
        ))),
    };
    match r {
        Ok(v) => send_json(w, req, 200, &v),
        Err(e) => send_error(w, req, &e),
    }
}

fn body(req: &Request) -> Result<Map<String, Value>, ApiError> {
    match serde_json::from_slice::<Value>(&req.body) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => Err(invalid("the request body must be a JSON object")),
        Err(e) => Err(invalid(format!("invalid JSON body: {e}"))),
    }
}

/// A JSON object with each key named once, discarding its values.
pub struct Keys;
impl<'de> Visitor<'de> for Keys {
    type Value = Keys;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Keys, A::Error> {
        let mut seen = std::collections::HashSet::new();
        while let Some(k) = m.next_key::<String>()? {
            m.next_value::<IgnoredAny>()?;
            if seen.contains(&k) {
                return Err(A::Error::custom(format!("{k} is named twice")));
            }
            seen.insert(k);
        }
        Ok(Keys)
    }
}
impl<'de> Deserialize<'de> for Keys {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Keys, D::Error> {
        d.deserialize_map(Keys)
    }
}

/// The body of a route that reads [`logit_bias`]: [`body`], and a 400 for a
/// `logit_bias` key named twice, which the parse into a map keeps only the
/// last of.
fn gen_body(req: &Request) -> Result<Map<String, Value>, ApiError> {
    /// The body's top level, which reads only its `logit_bias`.
    struct Top;
    impl<'de> Visitor<'de> for Top {
        type Value = ();
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a JSON object")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
            while let Some(k) = m.next_key::<String>()? {
                if k == "logit_bias" {
                    m.next_value::<Keys>()
                        .map_err(|e| A::Error::custom(format!("logit_bias: {e}")))?;
                } else {
                    m.next_value::<IgnoredAny>()?;
                }
            }
            Ok(())
        }
    }

    let b = body(req)?;
    if matches!(b.get("logit_bias"), Some(Value::Object(_))) {
        serde_json::Deserializer::from_slice(&req.body)
            .deserialize_map(Top)
            .map_err(|e| invalid(format!("invalid JSON body: {e}")).at("logit_bias"))?;
    }
    Ok(b)
}

// ---------------------------------------------------------------- field helpers

/// A field of another type than it takes: a 400 naming the field, the type it
/// takes and the type it got.
fn wrong_type(k: &str, takes: &str, v: &Value) -> ApiError {
    ::embed::request::wrong_type(k, takes, v).into()
}

/// A real field as `f32`. Absent or `null` is `None`, any other type than a
/// number a 400; a number whose `f32` is not finite (`1e39`, or `1e999` past
/// `f64`) is a 400 naming the field, as a penalty's is, where the sampler
/// would refuse it and the request would run greedy.
fn get_f32(o: &Map<String, Value>, k: &str) -> Result<Option<f32>, ApiError> {
    let Some(v) = o.get(k).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if !v.is_number() {
        return Err(wrong_type(k, "a number", v));
    }
    v.as_f64()
        .map(|x| x as f32)
        .filter(|x| x.is_finite())
        .map(Some)
        .ok_or_else(|| invalid(format!("{k} must be a finite number, not {v}")).at(k))
}

/// An integer field. Absent or `null` is `None`, any other type than a number
/// a 400; a number that is not an integer (`2.5`, or past `i64`) is a 400
/// naming the field, where llama-server would truncate it. An integer-valued
/// float (`2.0`) passes.
fn get_i(o: &Map<String, Value>, k: &str) -> Result<Option<i64>, ApiError> {
    ::embed::request::integer(o, k).map_err(ApiError::from)
}

/// An integer field within llama-server's hard limits `lo..=i32::MAX` (its
/// counts and token limit): absent or `null` is `None`; out of range is a 400
/// naming the field and the range, as a wrong type is.
fn get_i_in(o: &Map<String, Value>, k: &str, lo: i64) -> Result<Option<i64>, ApiError> {
    get_i(o, k)?.map(|n| in_limits(k, n, lo)).transpose()
}

/// `n` held to `lo..=i32::MAX` for the field `k`, else a 400 naming both.
fn in_limits(k: &str, n: i64, lo: i64) -> Result<i64, ApiError> {
    if (lo..=i64::from(i32::MAX)).contains(&n) {
        return Ok(n);
    }
    Err(invalid(format!(
        "{k} must be an integer in {lo}..={}, as llama-server takes it, not {n}",
        i32::MAX
    ))
    .at(k))
}

/// A boolean field. Absent or `null` is `None`, any other type than a boolean
/// a 400.
fn get_b(o: &Map<String, Value>, k: &str) -> Result<Option<bool>, ApiError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(wrong_type(k, "a boolean", v)),
    }
}

/// `stop` as a string or an array of strings; absent or `null` holds none, an
/// array element that is not a string and any other type is a 400 naming the
/// field.
fn stop_list(v: Option<&Value>) -> Result<Vec<String>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid(format!("stop must hold strings, not {x}")).at("stop"))
            })
            .collect(),
        Some(other) => Err(wrong_type("stop", "a string or an array of strings", other)),
    }
}

/// The llama.cpp samplers this server has none of: each is a 400 naming it
/// when its own field turns it on, `typical_p < 1`, `mirostat > 0` (llama-server
/// clamps it to `0..=2`), `dry_multiplier` other than 0, `xtc_probability > 0`,
/// `top_n_sigma > 0` or `dynatemp_range > 0` (the off ranges of llama.cpp's
/// `llama-sampler.cpp`). A secondary field can also leave a sampler off
/// (`dry_penalty_last_n` 0, `xtc_threshold > 0.5`); that does not lift the
/// refusal. The secondary fields pass at any value of the type they take, except
/// the values llama-server's schema refuses: `dry_allowed_length` or
/// `dry_penalty_last_n` below 0, and a `dry_sequence_breakers` that is not a
/// non-empty array of strings. `null` is unset.
fn unserved_samplers(o: &Map<String, Value>) -> Result<(), ApiError> {
    let on = [
        (
            "typical_p",
            get_f32(o, "typical_p")?.is_some_and(|p| p < 1.0),
        ),
        ("mirostat", get_i(o, "mirostat")?.is_some_and(|m| m > 0)),
        (
            "dry_multiplier",
            get_f32(o, "dry_multiplier")?.is_some_and(|m| m != 0.0),
        ),
        (
            "xtc_probability",
            get_f32(o, "xtc_probability")?.is_some_and(|p| p > 0.0),
        ),
        (
            "top_n_sigma",
            get_f32(o, "top_n_sigma")?.is_some_and(|n| n > 0.0),
        ),
        (
            "dynatemp_range",
            get_f32(o, "dynatemp_range")?.is_some_and(|r| r > 0.0),
        ),
    ];
    if let Some((field, _)) = on.iter().find(|(_, hit)| *hit) {
        return Err(invalid(format!("{field} is not supported by this server")).at(field));
    }
    for k in [
        "mirostat_tau",
        "mirostat_eta",
        "dry_base",
        "xtc_threshold",
        "dynatemp_exponent",
    ] {
        get_f32(o, k)?;
    }
    for k in ["dry_allowed_length", "dry_penalty_last_n"] {
        get_i_in(o, k, 0)?;
    }
    let takes = "a non-empty array of strings";
    match o.get("dry_sequence_breakers") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_string) => Ok(()),
        Some(Value::Array(_)) => Err(invalid(format!(
            "dry_sequence_breakers must be {takes}, as llama-server takes it"
        ))
        .at("dry_sequence_breakers")),
        Some(v) => Err(wrong_type("dry_sequence_breakers", takes, v)),
    }
}

/// The order this server samples in, by llama-server's sampler names; `/props`
/// and `generation_settings` report it.
const SAMPLERS: [&str; 5] = ["penalties", "top_k", "top_p", "min_p", "temperature"];

/// A request's `samplers`, which llama-server reads as the order its samplers
/// run in: an array of names (any ASCII case; its aliases `top-k`, `topk`, `nucleus`,
/// `temp`, `typ` and the like) or a string of one character a sampler. This
/// server runs one order, [`SAMPLERS`], so only that order is taken, after
/// every sampler that changes nothing at this request's values drops out of
/// both lists (llama-server's `dry`, `typ_p`, `top_n_sigma` and `xtc` always;
/// the penalties, `top_k`, `top_p` and `min_p` at their off values, and
/// `temperature` at 1); another
/// order, a name or character no sampler has (llama-server skips those with a
/// warning), or a sampler this server lacks is a 400 naming `samplers`. `null`
/// is unset.
fn sampler_order(o: &Map<String, Value>, sp: &SamplingParams) -> Result<(), ApiError> {
    const NAMES: [(&str, char); 11] = [
        ("dry", 'd'),
        ("top_k", 'k'),
        ("typ_p", 'y'),
        ("top_p", 'p'),
        ("top_n_sigma", 's'),
        ("min_p", 'm'),
        ("temperature", 't'),
        ("xtc", 'x'),
        ("infill", 'i'),
        ("penalties", 'e'),
        ("adaptive_p", 'a'),
    ];
    let by_name = |name: &str| {
        let name = name.to_ascii_lowercase();
        let alias = match name.as_str() {
            "nucleus" => "top_p",
            "temp" => "temperature",
            "typ" => "typ_p",
            other => other,
        };
        NAMES
            .iter()
            .map(|(canonical, _)| *canonical)
            .find(|canonical| {
                *canonical == alias
                    || canonical.replace('_', "-") == alias
                    || canonical.replace('_', "") == alias
            })
    };
    let by_char = |c: char| NAMES.iter().find(|(_, x)| *x == c).map(|(name, _)| *name);
    let order: Option<Vec<&str>> = match o.get("samplers") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().and_then(by_name)).collect(),
        Some(Value::String(s)) => s.chars().map(by_char).collect(),
        Some(v) => {
            return Err(wrong_type(
                "samplers",
                "an array of sampler names or a string of sampler characters",
                v,
            ));
        }
    };
    // A sampler that changes nothing at this request's values drops out of
    // both lists before they are compared, so its place in the order cannot
    // matter: dry, typ_p, top_n_sigma and xtc always (gen_params runs
    // unserved_samplers first, which refuses each one whose field turns it
    // on), the penalties when none applies, and top_k, top_p and min_p at
    // their off values. Temperature drops out only at 1, which divides by 1:
    // at `<= 0` it keeps only the best candidate, and llama-server without it
    // draws at temperature 1.
    // Under that greedy pick top_k, top_p and min_p change nothing either,
    // but only while no penalty runs, since a penalty before them can change
    // which candidate is best.
    let neutral = sp.temperature <= 0.0 && !sp.penalizes();
    let off = |name: &str| match name {
        "dry" | "typ_p" | "top_n_sigma" | "xtc" => true,
        "penalties" => !sp.penalizes(),
        "top_k" => neutral || sp.top_k <= 0,
        "top_p" => neutral || sp.top_p >= 1.0,
        "min_p" => neutral || sp.min_p <= 0.0,
        "temperature" => sp.temperature == 1.0,
        _ => false,
    };
    let on = |names: &[&'static str]| -> Vec<&'static str> {
        names.iter().copied().filter(|name| !off(name)).collect()
    };
    if order.as_deref().map(on) == Some(on(&SAMPLERS)) {
        return Ok(());
    }
    Err(invalid(format!(
        "samplers must be this server's order {SAMPLERS:?} (or \"ekpmt\"), not {}",
        o["samplers"]
    ))
    .at("samplers"))
}

/// Knobs shared by both generation endpoints. The token limit is the first set
/// of `n_predict`, `max_completion_tokens` and `max_tokens`, llama-server's
/// order on both routes (`n_predict` first, the OpenAI names its aliases in that
/// order); every one that is set must be an integer, the shadowed ones too, and
/// the one taken must be in `-1..=i32::MAX` (`-1` no limit), llama-server's hard
/// limits on `n_predict`; outside them it is a 400 naming that field.
/// `cache_prompt` (default `true`, as llama-server) keeps the longest prefix the
/// engine's cache already holds; `false` resets it first.
///
/// A field this server cannot honor is a 400 naming it, never a 200 that ignores it:
/// `n_probs > 0`, a non-empty `grammar`, `logprobs` other than `false` (a bool, or
/// OpenAI's text completion count), `top_logprobs > 0`, `n > 1` (the OpenAI
/// chat and text completion routes take it out first, [`take_choices`]), the samplers [`unserved_samplers`] names, and a `samplers` other
/// than this server's order ([`sampler_order`]). `null` counts as absent. Other
/// unknown fields are ignored. `logit_bias` is read by [`logit_bias`].
///
/// Structured output ([`constrain::request`]) is served for a `response_format`
/// of type `json_object`, or `json_schema` in the subset it reads, and for
/// `tool_choice` `"required"` or a named function over the request's tools;
/// any other `response_format`, `json_schema`, `tool_choice` or schema keyword
/// is a 400 naming it. Such a request's `temperature` defaults to 0 (greedy); a
/// positive one and the penalties draw with the sampler among the ids the
/// program allows, and `ignore_eos` and `reasoning_budget` are 400s. The two
/// routes whose answers render probabilities take their fields out of the body
/// before this runs
/// ([`completion_plan_logprobs`], [`chat_plan_logprobs`]); every other route
/// keeps the refusal.
///
/// The penalties take llama-server's names and defaults (`repeat_penalty` 1,
/// `frequency_penalty` 0, `presence_penalty` 0, `repeat_last_n` 64) and
/// llama.cpp's checks (`common_sampler_init`): a non-finite penalty, or a
/// `repeat_penalty` that is not `> 0` with a finite reciprocal, is a 400
/// naming it. `repeat_last_n` outside llama-server's `0..=i32::MAX` is a 400. A
/// greedy request takes the engine's argmax, which no penalty reaches, so
/// penalties that change a logit with `temperature <= 0` are a 400. A
/// `temperature`, `top_p` or `min_p` whose `f32` is not finite (`1e39`) is a
/// 400 naming it; the sampler's other ranges stay as it takes them.
fn gen_params(state: &State, o: &Map<String, Value>) -> Result<GenParams, ApiError> {
    let set = |k: &str| o.get(k).filter(|v| !v.is_null());
    let refused = [
        ("n_probs", get_i(o, "n_probs")?.is_some_and(|n| n > 0)),
        (
            "grammar",
            set("grammar").is_some_and(|v| v.as_str() != Some("")),
        ),
        // A bool on the chat path, a count on OpenAI's text completion path:
        // any value but `false` asks for probabilities.
        (
            "logprobs",
            set("logprobs").is_some_and(|v| v.as_bool() != Some(false)),
        ),
        (
            "top_logprobs",
            get_i(o, "top_logprobs")?.is_some_and(|n| n > 0),
        ),
        ("n", get_i(o, "n")?.is_some_and(|n| n > 1)),
    ];
    if let Some((field, _)) = refused.iter().find(|(_, hit)| *hit) {
        return Err(invalid(format!("{field} is not supported by this server")).at(field));
    }
    unserved_samplers(o)?;
    let constraint = constrain::request(
        o,
        &constrain::Ask {
            tok: &*state.tok,
            vocab: &state.vocab,
            format: state.tool_format,
            think: state.think,
        },
    )
    .map_err(|r| {
        if r.unsupported {
            ApiError {
                retry_after: false,
                param: None,
                code: 501,
                kind: "not_supported_error",
                message: r.message,
            }
        } else {
            let e = invalid(r.message);
            match &r.field {
                Some(f) => e.at(f),
                None => e,
            }
        }
    })?;
    let d = SamplingParams::default();
    let seed = match get_i(o, "seed")? {
        Some(s) if s >= 0 => s as u64,
        _ => mix(state.next_id() ^ unix_now().rotate_left(17)) & u64::from(u32::MAX),
    };
    let limits = [
        ("n_predict", get_i(o, "n_predict")?),
        ("max_completion_tokens", get_i(o, "max_completion_tokens")?),
        ("max_tokens", get_i(o, "max_tokens")?),
    ];
    let n_predict = match limits.into_iter().find_map(|(k, n)| n.map(|n| (k, n))) {
        None => -1,
        Some((k, n)) => in_limits(k, n, -1)?,
    };
    let top_k = get_i(o, "top_k")?;
    let penalty = |k: &str, d: f32, ok: fn(f32) -> bool, rule: &str| match set(k) {
        None => Ok(d),
        Some(v) => v
            .as_f64()
            .map(|x| x as f32)
            .filter(|&x| ok(x))
            .ok_or_else(|| invalid(format!("{k} must be {rule}, not {v}")).at(k)),
    };
    let finite = "a finite number";
    let repeat_last_n = get_i_in(o, "repeat_last_n", 0)?.map_or(d.repeat_last_n, |n| {
        usize::try_from(n).expect("a count in 0..=i32::MAX fits usize")
    });
    let sampling = SamplingParams {
        temperature: get_f32(o, "temperature")?.unwrap_or(if constraint.is_some() {
            0.0
        } else {
            d.temperature
        }),
        top_k: top_k.map_or(d.top_k, |k| i32::try_from(k).unwrap_or(i32::MAX)),
        top_p: get_f32(o, "top_p")?.unwrap_or(d.top_p),
        min_p: get_f32(o, "min_p")?.unwrap_or(d.min_p),
        repeat_penalty: penalty(
            "repeat_penalty",
            d.repeat_penalty,
            |r| r.is_finite() && r > 0.0 && (1.0 / r).is_finite(),
            "a finite number > 0 with a finite reciprocal",
        )?,
        frequency_penalty: penalty(
            "frequency_penalty",
            d.frequency_penalty,
            f32::is_finite,
            finite,
        )?,
        presence_penalty: penalty(
            "presence_penalty",
            d.presence_penalty,
            f32::is_finite,
            finite,
        )?,
        repeat_last_n,
        seed,
    };
    sampler_order(o, &sampling)?;
    let ignore_eos = get_b(o, "ignore_eos")?.unwrap_or(false);
    let reasoning_budget = reasoning_budget(o)?;
    let logit_bias = logit_bias(o, state.info.n_vocab)?;
    if constraint.is_some() {
        program_ends_itself(ignore_eos, reasoning_budget)?;
    }
    let p = GenParams {
        n_predict,
        sampling,
        stop: stop_list(o.get("stop"))?,
        ignore_eos,
        stream: get_b(o, "stream")?.unwrap_or(false),
        timings_per_token: get_b(o, "timings_per_token")?.unwrap_or(false),
        return_progress: get_b(o, "return_progress")?.unwrap_or(false),
        include_usage: o
            .get("stream_options")
            .and_then(|s| s.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        cache_prompt: get_b(o, "cache_prompt")?.unwrap_or(true),
        reasoning_budget,
        reasoning_loop: get_b(o, "reasoning_loop")?.unwrap_or(false),
        think_entry: ThinkEntry::Closed,
        think: state.think,
        think_split: None,
        logprobs: None,
        constraint,
        logit_bias,
    };
    Ok(p)
}

/// OpenAI's `logit_bias`: an object from token ids (decimal strings) to
/// numbers in `-100..=100`, each added to its id's logit before every pick,
/// on greedy, sampled and structured requests alike. Absent, `null`, `{}`
/// and `[]` change nothing, and a zero bias is dropped. llama-server's other
/// forms (an array of pairs, `false` or a token's text), a key that is not an
/// id below `n_vocab`, a value out of range and one id named twice (`"7"`
/// and `"07"`, or `"7"` twice, which [`gen_body`] finds) are 400s naming the
/// field.
fn logit_bias(o: &Map<String, Value>, n_vocab: usize) -> Result<Vec<(u32, f32)>, ApiError> {
    let m = match o.get("logit_bias") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) if a.is_empty() => return Ok(Vec::new()),
        Some(Value::Object(m)) => m,
        Some(v) => {
            return Err(invalid(format!(
                "logit_bias must be an object of token ids to numbers in -100..=100, not {v}"
            ))
            .at("logit_bias"));
        }
    };
    let mut bias = std::collections::BTreeMap::new();
    for (k, v) in m {
        let id = k
            .parse::<u32>()
            .ok()
            .filter(|&id| {
                k.bytes().all(|c| c.is_ascii_digit())
                    && usize::try_from(id).is_ok_and(|i| i < n_vocab)
            })
            .ok_or_else(|| {
                invalid(format!(
                    "logit_bias key {k:?} is not a token id below the vocabulary's {n_vocab}"
                ))
                .at("logit_bias")
            })?;
        let b = v
            .as_f64()
            .filter(|b| (-100.0..=100.0).contains(b))
            .ok_or_else(|| {
                invalid(format!(
                    "logit_bias[{k:?}] must be a number in -100..=100, not {v}"
                ))
                .at("logit_bias")
            })?;
        if bias.insert(id, b as f32).is_some() {
            return Err(invalid(format!("logit_bias names token id {id} twice")).at("logit_bias"));
        }
    }
    Ok(bias.into_iter().filter(|&(_, b)| b != 0.0).collect())
}

/// Whether `p`'s program is a forced tool call that starts after the model's
/// own close of an open think span ([`constrain::Constraint::follows_reasoning`]).
fn follows_reasoning(p: &GenParams) -> bool {
    p.constraint
        .as_ref()
        .is_some_and(constrain::Constraint::follows_reasoning)
}

/// What a structured-output request leaves out: `ignore_eos` would ban the id
/// its program ends on, and a think budget would force a close that competes
/// with the program for the span's end: a JSON document closes an open span
/// itself, and a forced tool call starts at the model's own close.
fn program_ends_itself(ignore_eos: bool, budget: Option<usize>) -> Result<(), ApiError> {
    let (field, why) = if ignore_eos {
        (
            "ignore_eos",
            "ignore_eos is not supported with structured output: its program ends on an \
             end-of-generation id",
        )
    } else if budget.is_some() {
        (
            "reasoning_budget",
            "reasoning_budget is not supported with structured output: a JSON response closes \
             an open think span itself, and a forced tool call starts at the model's own close",
        )
    } else {
        return Ok(());
    };
    Err(invalid(why).at(field))
}

/// `reasoning_budget`: absent or `null` unrestricted, `-1` unrestricted
/// (llama-server's spelling), an integer `N >= 0` the think span's budget in
/// generated ids, any other integer a 400 naming it, a non-integer a 400
/// naming its type. An integer-valued float (`8.0`) passes, as every integer
/// field's does.
fn reasoning_budget(o: &Map<String, Value>) -> Result<Option<usize>, ApiError> {
    let Some(v) = o.get("reasoning_budget").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let n = match v {
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64().and_then(|f| {
                (f.fract() == 0.0 && f >= 0.0 && f <= i64::MAX as f64).then_some(f as i64)
            })
        }),
        _ => None,
    };
    match n {
        Some(-1) => Ok(None),
        Some(n) if n >= 0 => Ok(Some(usize::try_from(n).unwrap_or(usize::MAX))),
        Some(n) => Err(invalid(format!(
            "reasoning_budget {n} is not -1 (unrestricted) or an integer >= 0"
        ))
        .at("reasoning_budget")),
        None => Err(invalid(format!(
            "reasoning_budget must be an integer, not {}",
            json_type(v)
        ))
        .at("reasoning_budget")),
    }
}

/// A JSON value's type, for a field's refusal naming what it got.
use ::embed::request::json_type;

/// The think-span budget and the loop guard live only where a span can open,
/// `p.think`'s span (the server's tags): a prompt that ends with the start tag
/// starts the model inside it, one that ends with the closed span (the
/// template with thinking off) has no reasoning to cap or guard, so both are
/// silently ignored there — llama-server ignores `--reasoning-budget` when
/// thinking is already off by other means — and one that ends with neither
/// leaves the span to the model, where the budget starts counting and the
/// guard starts reading at the start tag, as llama-server's sampler does. The
/// entry lands in `p` for the generation's tracker whatever the budget: the
/// ids before a model-opened span spend nothing, and its close ids are not
/// forced until the span opens.
fn gate_think_span(p: &mut GenParams, prompt: &str) {
    p.think_entry = ThinkEntry::of_prompt_with(prompt, p.think);
    if p.think_entry == ThinkEntry::Closed {
        p.reasoning_budget = None;
        p.reasoning_loop = false;
    }
}

/// llama-server's `generation_settings`, with the values this server applies.
/// Knobs it does not implement are reported at their neutral setting.
/// Its f32 values are widened to f64 before JSON serialization.
fn generation_settings(state: &State, p: &GenParams) -> Value {
    let logit_bias: Vec<Value> = p
        .logit_bias
        .iter()
        .map(|&(token, bias)| json!({"bias": f64::from(bias), "token": token}))
        .collect();
    let mut v = json!({
        "n_ctx": state.info.ctx_max,
        "n_predict": p.n_predict,
        "model": state.alias,
        "seed": p.sampling.seed,
        "temperature": f64::from(p.sampling.temperature),
        "dynatemp_range": 0.0,
        "dynatemp_exponent": 1.0,
        "top_k": p.sampling.top_k,
        "top_p": f64::from(p.sampling.top_p),
        "min_p": f64::from(p.sampling.min_p),
        "tfs_z": 1.0,
        "typical_p": 1.0,
        "repeat_last_n": p.sampling.repeat_last_n,
        "repeat_penalty": f64::from(p.sampling.repeat_penalty),
        "presence_penalty": f64::from(p.sampling.presence_penalty),
        "frequency_penalty": f64::from(p.sampling.frequency_penalty),
        "penalty_prompt_tokens": [],
        "use_penalty_prompt_tokens": false,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "penalize_nl": false,
        "stop": p.stop,
        "max_tokens": p.n_predict,
        "n_keep": 0,
        "n_discard": 0,
        "ignore_eos": p.ignore_eos,
        "stream": p.stream,
        "logit_bias": logit_bias,
        "n_probs": p.logprobs.as_ref().map_or(0, logprobs::Ask::asked),
        "min_keep": 0,
        "grammar": "",
        "samplers": SAMPLERS,
    });
    // The samplers [`unserved_samplers`] refuses, at llama-server's off values
    // (one `json!` over every key passes the macro's recursion limit).
    for (k, off) in [
        ("top_n_sigma", json!(-1.0)),
        ("xtc_probability", json!(0.0)),
        ("xtc_threshold", json!(0.1)),
        ("dry_multiplier", json!(0.0)),
        ("dry_base", json!(1.75)),
        ("dry_allowed_length", json!(2)),
        ("dry_penalty_last_n", json!(64)),
        ("dry_sequence_breakers", json!(["\n", ":", "\"", "*"])),
    ] {
        v[k] = off;
    }
    v
}

/// The settings a request with an empty body would get (seed shown as llama-server's default).
fn default_params(state: &State) -> GenParams {
    GenParams {
        n_predict: -1,
        sampling: SamplingParams::default(),
        stop: Vec::new(),
        ignore_eos: false,
        stream: false,
        timings_per_token: false,
        return_progress: false,
        include_usage: false,
        cache_prompt: true,
        reasoning_budget: None,
        reasoning_loop: false,
        think_entry: ThinkEntry::Closed,
        think: state.think,
        think_split: None,
        logprobs: None,
        constraint: None,
        logit_bias: Vec::new(),
    }
}

// ---------------------------------------------------------------- read-only endpoints

fn health(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    if let Some(reason) = state.fatal() {
        return send_json(w, req, 503, &fatal_health(&reason));
    }
    let idle = relock(&state.shared.board).free();
    let v = json!({
        "status": if idle == 0 { "no slot available" } else { "ok" },
        "slots_idle": idle,
        "slots_processing": state.parallel - idle,
    });
    let status = if idle == 0 && req.has_query("fail_on_no_slot") {
        503
    } else {
        200
    };
    send_json(w, req, status, &v)
}

/// The engine's model, then the side model when the server holds one.
fn models(state: &State) -> Value {
    let mut models = vec![Listed {
        id: &state.alias,
        meta: json!({ "n_vocab": state.info.n_vocab, "n_ctx_train": state.info.ctx_max }),
        max_model_len: Some(state.info.ctx_max),
    }];
    if let Some(side) = &state.side {
        models.push(Listed {
            id: &side.name,
            meta: json!({ "n_ctx": side.ctx, "n_embd": side.width(), "prompt_names": side.prompts.keys().collect::<Vec<_>>() }),
            max_model_len: Some(side.ctx),
        });
    }
    crate::models::listing(models, state.start_unix)
}

/// Ollama's `POST /api/show` of a model [`models`] lists, named by `model` (or Ollama's older
/// `name`); another name is the lookup's 404. The engine's model answers its chat template,
/// what the file states of its format, architecture and quant, the served context as
/// `<arch>.context_length`, and its `capabilities`: `completion`, `tools` when the template
/// teaches a tool-call markup the server parses, `vision` with an image input. The side model
/// answers `embedding` alone.
fn show(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    let field = |k| b.get(k).filter(|v| !v.is_null());
    let id = match field("model").or_else(|| field("name")) {
        Some(Value::String(s)) => s,
        Some(v) => return Err(wrong_type("model", "a string", v)),
        None => return Err(invalid("model is required: a name /api/tags lists")),
    };
    let entry = crate::models::lookup(&models(state), id).map_err(not_found)?;
    if entry["id"] != state.alias.as_str() {
        return Ok(json!({ "details": {}, "model_info": {}, "capabilities": ["embedding"] }));
    }
    let file = &state.engine_props["model"];
    let (mut details, mut info) = (Map::new(), Map::new());
    put(&mut details, "format", file["format"].as_str());
    put(&mut details, "quantization_level", file["quant"].as_str());
    if let Some(arch) = file["arch"].as_str() {
        details.insert("family".into(), json!(arch));
        details.insert("families".into(), json!([arch]));
        info.insert(gguf::GENERAL_ARCHITECTURE.into(), json!(arch));
        info.insert(format!("{arch}.context_length"), json!(state.info.ctx_max));
    }
    let mut capabilities = vec!["completion"];
    if !matches!(state.tool_format, ToolFormat::Unparsed) {
        capabilities.push("tools");
    }
    if state.media.is_some() {
        capabilities.push("vision");
    }
    Ok(json!({
        "template": state.template.source(),
        "details": details,
        "model_info": info,
        "capabilities": capabilities,
    }))
}

fn props(state: &State) -> Value {
    let mut props = json!({
        "system_prompt": "",
        "model_alias": state.alias,
        "model_path": state.model_path,
        "model_name": state.alias,
        "default_generation_settings": generation_settings(state, &default_params(state)),
        "total_slots": state.parallel,
        "chat_template": state.template.source(),
        "thinking_off": state.thinking_off.word(),
        "chat_template_caps": {},
        "bos_token": state.info.bos_text,
        "eos_token": state.info.eos_text,
        "modalities": {
            "vision": state.media.is_some(),
            "audio": false,
            "video": state.takes_video(),
        },
        "n_ctx": state.info.ctx_max,
        "routes": systemone::ROUTES,
        "build_info": format!("bloomery-serve {VERSION}"),
        "engine": state.engine_props,
    });
    if let Some(side) = &state.side {
        props["embedding_model"] = json!({ "alias": side.name, "path": side.path });
    }
    props
}

/// The commit this build names, set by `build.rs`. `--version` reads it too.
pub const COMMIT: &str = env!("BLOOMERY_SERVE_COMMIT");

/// `engine.version`: the crate version and [`COMMIT`].
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("BLOOMERY_SERVE_COMMIT"),
    ")"
);

/// Inserts `key` when `v` is `Some`: an unknown is left out, never guessed.
fn put<T: Into<Value>>(o: &mut Map<String, Value>, key: &str, v: Option<T>) {
    if let Some(v) = v {
        o.insert(key.to_owned(), v.into());
    }
}

/// The process's argv as `/props`' `engine.args` gives it: verbatim but for
/// the token after `--api-key` — the key itself, which no answer may carry —
/// replaced by `<redacted>`. The seats take the two-token spelling (a flag,
/// its value), so that is the one replaced; `--api-key-file`'s token names a
/// file, not a key, and stays.
fn argv_redacted(args: impl Iterator<Item = String>) -> Vec<String> {
    let mut redact = false;
    args.map(|a| {
        let out = if redact {
            "<redacted>".to_owned()
        } else {
            a.clone()
        };
        redact = a == "--api-key";
        out
    })
    .collect()
}

/// `/props`' `engine` object in toktape's shape: the server's own `name`,
/// `version` (with the engine's note), `args` (this process's argv, verbatim
/// but for the key [`argv_redacted`] holds back) and `server_pid`, and the
/// model, placement, draft and verified context the engine reports.
pub(crate) fn engine_object(p: &EngineProps) -> Value {
    let mut o = Map::new();
    o.insert("name".into(), json!("bloomery"));
    let version = match &p.version_note {
        Some(note) => format!("{VERSION} {note}"),
        None => VERSION.to_owned(),
    };
    o.insert("version".into(), Value::String(version));
    let args = argv_redacted(std::env::args_os().map(|a| a.to_string_lossy().into_owned()))
        .into_iter()
        .map(Value::String)
        .collect();
    o.insert("args".into(), Value::Array(args));
    put(&mut o, "model", p.model.as_ref().map(model_object));
    put(
        &mut o,
        "placement",
        p.placement.as_ref().map(placement_object),
    );
    put(&mut o, "draft", p.draft.as_ref().map(draft_object));
    put(&mut o, "ctx_verified", p.ctx_verified);
    o.insert("server_pid".into(), json!(std::process::id()));
    Value::Object(o)
}

fn model_object(m: &ModelProps) -> Value {
    let mut o = Map::new();
    put(&mut o, "format", m.format.clone());
    put(&mut o, "arch", m.arch.clone());
    put(&mut o, "quant", m.quant.clone());
    put(&mut o, "bytes", m.bytes);
    put(&mut o, "files", m.files);
    put(&mut o, "n_layers", m.n_layers);
    put(&mut o, "n_experts", m.n_experts);
    put(&mut o, "n_experts_used", m.n_experts_used);
    put(&mut o, "ctx_train", m.ctx_train);
    Value::Object(o)
}

/// Each device's `bytes` is the sum of its classes, here and nowhere else.
fn placement_object(p: &PlacementProps) -> Value {
    let devices = p
        .devices
        .iter()
        .map(|d| {
            let mut o = Map::new();
            o.insert("device".into(), Value::String(d.device.clone()));
            let bytes = d
                .class_bytes
                .values()
                .fold(0u64, |sum, &b| sum.saturating_add(b));
            o.insert("bytes".into(), json!(bytes));
            let classes = d
                .class_bytes
                .iter()
                .map(|(k, &b)| (k.clone(), json!(b)))
                .collect();
            o.insert("classes".into(), Value::Object(classes));
            put(&mut o, "layers", d.layers.clone());
            Value::Object(o)
        })
        .collect();
    let mut o = Map::new();
    o.insert("devices".into(), Value::Array(devices));
    put(&mut o, "vram_kv_bytes", p.vram_kv_bytes);
    Value::Object(o)
}

fn draft_object(d: &DraftProps) -> Value {
    let mut o = Map::new();
    o.insert("model".into(), Value::String(d.model.clone()));
    put(&mut o, "n_max", d.n_max);
    put(&mut o, "kind", d.kind.clone());
    put(&mut o, "path", d.path.clone());
    put(&mut o, "device", d.device.clone());
    Value::Object(o)
}

fn slots(state: &State) -> Value {
    let turns = relock(&state.shared.turns).clone();
    let b = relock(&state.shared.board);
    let list = b
        .slots()
        .iter()
        .enumerate()
        .map(|(id, slot)| {
            let busy = slot.state != Use::Free;
            let s = &slot.view;
            // llama-server's count of tokens left: only a bounded request that is running has one.
            let n_remain = if busy && s.n_predict >= 0 {
                let done = i64::try_from(s.n_decoded).unwrap_or(i64::MAX);
                s.n_predict.saturating_sub(done).max(0)
            } else {
                -1
            };
            let mut v = match &s.settings {
                Value::Object(_) => s.settings.clone(),
                _ => generation_settings(state, &default_params(state)),
            };
            if let Value::Object(m) = &mut v {
                m.insert("id".into(), json!(id));
                m.insert("id_task".into(), json!(s.id_task));
                m.insert("task_id".into(), json!(s.id_task));
                m.insert("state".into(), json!(i32::from(busy)));
                m.insert("is_processing".into(), json!(busy));
                m.insert("n_past".into(), json!(s.n_past));
                if let Some(t) = turns.as_ref().and_then(|t| t.get(id)) {
                    m.insert("turn".into(), json!(t.as_str()));
                }
                m.insert("prompt".into(), s.prompt.clone());
                m.insert(
                    "next_token".into(),
                    json!({
                        "has_next_token": busy,
                        "n_remain": n_remain,
                        "n_decoded": s.n_decoded,
                        "stopped_eos": s.stopped_eos,
                        "stopped_word": s.stopped_word,
                        "stopped_limit": s.stopped_limit,
                        "stopping_word": s.stopping_word,
                    }),
                );
            }
            v
        })
        .collect();
    Value::Array(list)
}

fn metrics(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let (busy, deferred, n_past) = {
        let b = relock(&state.shared.board);
        (state.parallel - b.free(), b.waiting(), b.n_past())
    };
    let room = state.room();
    let kv_ratio = if room > 0 {
        n_past as f64 / room as f64
    } else {
        0.0
    };
    let turns = relock(&state.shared.turns).is_some();
    let text = {
        let s = relock(&state.shared.stats);
        let per_decode = if s.n_decode_total > 0 {
            s.n_busy_slots_total as f64 / s.n_decode_total as f64
        } else {
            0.0
        };
        let rows: [(&str, &str, &str, String); 21] = [
            (
                "counter",
                "prompt_tokens_total",
                "Number of prompt tokens processed, excluding cached tokens",
                s.n_prompt_total.to_string(),
            ),
            (
                "counter",
                "prompt_tokens_cached_total",
                "Number of prompt tokens reused from the cache",
                s.n_prompt_cached_total.to_string(),
            ),
            (
                "counter",
                "prompt_seconds_total",
                "Prompt process time",
                (s.t_prompt_ms_total / 1e3).to_string(),
            ),
            (
                "counter",
                "prompt_cache_seconds_total",
                "Prompt cache time before the prompts (the requests' cache_ms: a slot state saved, \
                 a cached state put back, the cut); bloomery's own, llama-server has no such counter",
                (s.t_cache_ms_total / 1e3).to_string(),
            ),
            (
                "counter",
                "tokens_predicted_total",
                "Number of generation tokens processed.",
                s.n_predicted_total.to_string(),
            ),
            (
                "counter",
                "tokens_predicted_seconds_total",
                "Predict process time",
                (s.t_predicted_ms_total / 1e3).to_string(),
            ),
            (
                "counter",
                "n_decode_total",
                "Total number of llama_decode() calls",
                s.n_decode_total.to_string(),
            ),
            (
                "counter",
                "card_wait_seconds_total",
                "Time the engine waited for the card's work, every call's; bloomery's own",
                s.waits.card.as_secs_f64().to_string(),
            ),
            (
                "counter",
                "ple_wait_seconds_total",
                "Time the engine waited for PLE table rows read from storage, every call's; \
                 bloomery's own",
                s.waits.ple.as_secs_f64().to_string(),
            ),
            (
                "counter",
                "n_tokens_max",
                "Largest observed sequence length (prompt + generation)",
                s.n_tokens_max.to_string(),
            ),
            (
                "counter",
                "spec_decode_num_draft_tokens_total",
                "Speculative: Total draft tokens generated",
                s.n_draft_total.to_string(),
            ),
            (
                "counter",
                "spec_decode_num_accepted_tokens_total",
                "Speculative: Total draft tokens accepted by the target model",
                s.n_draft_accepted_total.to_string(),
            ),
            (
                "counter",
                "spec_decode_num_drafts_total",
                "Speculative: Total speculative decoding verification steps",
                s.n_draft_passes_total.to_string(),
            ),
            (
                "gauge",
                "n_busy_slots_per_decode",
                "Average number of busy slots per llama_decode() call",
                per_decode.to_string(),
            ),
            (
                "gauge",
                "prompt_tokens_seconds",
                "Average prompt throughput in tokens/s.",
                s.prompt_rate().to_string(),
            ),
            (
                "gauge",
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s.",
                s.decode_rate().to_string(),
            ),
            (
                "gauge",
                "kv_cache_usage_ratio",
                "KV-cache usage. 1 means 100 percent usage.",
                kv_ratio.to_string(),
            ),
            (
                "gauge",
                "kv_cache_tokens",
                "KV-cache tokens.",
                n_past.to_string(),
            ),
            (
                "gauge",
                "requests_processing",
                "Number of request processing.",
                busy.to_string(),
            ),
            (
                "counter",
                "requests_cancelled_before_prompt_total",
                "Requests cancelled before their prompt started",
                s.n_cancelled_before_prompt_total.to_string(),
            ),
            (
                "gauge",
                "requests_deferred",
                "Number of request deferred.",
                deferred.to_string(),
            ),
        ];
        let swap: [(&str, &str, &str, String); 7] = [
            (
                "counter",
                "swaps_total",
                "Slots that take the engine in turns: switches that moved a state",
                s.n_swaps_total.to_string(),
            ),
            (
                "counter",
                "swap_seconds_total",
                "Slots that take the engine in turns: time spent parking and putting back states",
                (s.t_swap_ms_total / 1e3).to_string(),
            ),
            (
                "counter",
                "swap_refusals_total",
                "Requests refused because the running request's state could not be parked",
                s.n_swap_refused_total.to_string(),
            ),
            (
                "counter",
                "swap_reprefill_tokens_total",
                "Positions a parked request's turn fed again (an engine that cannot snapshot)",
                s.n_reprefill_total.to_string(),
            ),
            (
                "counter",
                "swap_reprefill_seconds_total",
                "Time spent feeding parked requests' positions again",
                (s.t_reprefill_ms_total / 1e3).to_string(),
            ),
            (
                "gauge",
                "swap_parked_bytes",
                "Bytes of the parked states",
                s.parked_bytes.to_string(),
            ),
            (
                "gauge",
                "swap_park_budget_bytes",
                "The most bytes the parked states may hold (0: the engine parks ids, no state)",
                s.park_budget.to_string(),
            ),
        ];
        let held = state.images.stats();
        let images: [(&str, &str, &str, String); 4] = [
            (
                "counter",
                "prepared_image_hits_total",
                "Request images served from the prepared-image cache, without a decode or a prepare",
                held.hits.to_string(),
            ),
            (
                "counter",
                "prepared_image_misses_total",
                "Request images the prepared-image cache did not hold: decoded and prepared",
                held.misses.to_string(),
            ),
            (
                "gauge",
                "prepared_image_bytes",
                "Bytes the prepared-image cache retains",
                held.bytes.to_string(),
            ),
            (
                "gauge",
                "prepared_images",
                "Images the prepared-image cache retains",
                held.entries.to_string(),
            ),
        ];
        let vision = state.media.is_some();
        let mut out = String::new();
        for (kind, name, help, value) in rows
            .into_iter()
            .chain(swap.into_iter().filter(|_| turns))
            .chain(images.into_iter().filter(|_| vision))
        {
            out.push_str(&format!(
                "# HELP llamacpp:{name} {help}\n# TYPE llamacpp:{name} {kind}\nllamacpp:{name} {value}\n"
            ));
        }
        labelled_metrics(&mut out, &s, &state.engine_props);
        out
    };
    http::respond(
        w,
        req,
        200,
        "text/plain; version=0.0.4",
        &[("Process-Start-Time-Unix", state.start_unix.to_string())],
        text.as_bytes(),
    )?;
    Ok(true)
}

/// `/metrics`' counters split by a label, bloomery's own: the decode rounds
/// by width (their time and the engine's waits inside
/// them), and the draft's counts by depth and source (the drafter's kind,
/// `/props`' `engine.draft.kind`), written to `out`. A width no round ran
/// has no row.
fn labelled_metrics(out: &mut String, s: &Stats, engine_props: &Value) {
    let source = engine_props["draft"]["kind"].as_str().unwrap_or("unknown");
    let source = source
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    let rounds: Vec<(String, &Round)> = (1..)
        .zip(&s.rounds)
        .filter(|(_, r)| r.n > 0)
        .map(|(w, r)| (format!("width=\"{w}\""), r))
        .collect();
    let by_width = |value: fn(&Round) -> String| -> Vec<(String, String)> {
        rounds.iter().map(|(l, r)| (l.clone(), value(r))).collect()
    };
    let by_depth = |value: fn(&Depth) -> u64| -> Vec<(String, String)> {
        (1..)
            .zip(&s.draft_depths)
            .map(|(d, at)| {
                (
                    format!("source=\"{source}\",depth=\"{d}\""),
                    value(at).to_string(),
                )
            })
            .collect()
    };
    let families = [
        (
            "rounds_total",
            "Decode rounds by width, the slots a round stepped or passed; bloomery's own",
            by_width(|r| r.n.to_string()),
        ),
        (
            "round_seconds_total",
            "Decode round wall time by width: the engine's calls and the server's work \
             between them; bloomery's own",
            by_width(|r| (r.ms / 1e3).to_string()),
        ),
        (
            "round_card_wait_seconds_total",
            "Time the engine waited for the card's work inside decode rounds, by width; \
             bloomery's own",
            by_width(|r| r.waits.card.as_secs_f64().to_string()),
        ),
        (
            "round_ple_wait_seconds_total",
            "Time the engine waited for PLE table rows inside decode rounds, by width; \
             bloomery's own",
            by_width(|r| r.waits.ple.as_secs_f64().to_string()),
        ),
        (
            "spec_decode_num_draft_tokens_by_depth_total",
            "Speculative: draft tokens proposed at each depth of a draft; bloomery's own",
            by_depth(|d| d.proposed),
        ),
        (
            "spec_decode_num_accepted_tokens_by_depth_total",
            "Speculative: draft tokens accepted at each depth of a draft; bloomery's own",
            by_depth(|d| d.accepted),
        ),
    ];
    for (name, help, samples) in families {
        out.push_str(&format!(
            "# HELP llamacpp:{name} {help}\n# TYPE llamacpp:{name} counter\n"
        ));
        for (labels, value) in samples {
            out.push_str(&format!("llamacpp:{name}{{{labels}}} {value}\n"));
        }
    }
}

// ---------------------------------------------------------------- tokenizer endpoints

/// `/tokenize`. `content` is a string or an array of strings and token ids,
/// read as a `/completion` prompt is ([`prompt_ids`]); `add_special` gives it
/// the tokenizer's special tokens where it starts with text, as llama-server's
/// `tokenize_mixed`. An absent `content` is a 400
/// naming it (llama-server answers no tokens; an undefined input is an error
/// here), and so is any other type, `null` included.
fn tokenize(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    let add_special = get_b(b, "add_special")?.unwrap_or(false);
    let with_pieces = get_b(b, "with_pieces")?.unwrap_or(false);
    let t = &*state.tok;
    let ids = match b.get("content") {
        None => return Err(invalid("'content' is required").at("content")),
        Some(v) => prompt_ids(t, "content", v, add_special)?,
    };
    let tokens: Vec<Value> = if with_pieces {
        ids.iter()
            .map(|&id| json!({ "id": id, "piece": t.decode(&[id]) }))
            .collect()
    } else {
        ids.iter().map(|&id| json!(id)).collect()
    };
    Ok(json!({ "tokens": tokens }))
}

/// `/detokenize`: `tokens` is an array of token ids; an absent one is a 400
/// naming it.
fn detokenize(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    let ids = match b.get("tokens") {
        None => return Err(invalid("'tokens' is required").at("tokens")),
        Some(v) => token_ids(&*state.tok, v).map_err(|e| e.at("tokens"))?,
    };
    Ok(json!({ "content": state.tok.decode(&ids) }))
}

fn token_ids(tok: &dyn Tokenizer, v: &Value) -> Result<Vec<u32>, ApiError> {
    ::embed::input::token_ids(tok.n_vocab(), v).map_err(ApiError::from)
}

/// `/apply-template`: the rendered text, with its guarded token ids when requested.
/// Media placeholders stay unexpanded, as they do in the text.
fn apply_template(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    let return_ids = get_b(b, "return_prompt_ids")?.unwrap_or(false);
    let rendered = render_chat(state, b)?;
    let ids = return_ids.then(|| rendered.ids(&*state.tok));
    let mut reply = json!({ "prompt": rendered.text });
    if let Some(ids) = ids {
        reply["prompt_ids"] = json!(ids);
    }
    Ok(reply)
}

/// A chat template's render: its text, each image as the model's
/// placeholder and each clip as its text with a placeholder a group, and the
/// media in the order of their placeholders. The text is what the client sent
/// and the template wrote; `guarded` is the render the tokenizer reads when a
/// client string carried a special-token spelling ([`crate::guard`]).
struct Rendered {
    text: String,
    guarded: Option<Guarded>,
    media: Vec<Medium>,
}

impl Rendered {
    fn ids(&self, tok: &dyn Tokenizer) -> Vec<u32> {
        match &self.guarded {
            Some(guarded) => guarded.encode(tok),
            None => tok.encode(&self.text),
        }
    }
}

/// Renders the chat template for an OpenAI `messages` body. An array content
/// becomes one text: on an engine that takes images, its parts flattened by
/// the model's rule ([`media::flatten`]), each image the model's placeholder,
/// numbered by its place among the request's images, across messages
/// (`media`, in the order of the placeholders, holds them), and each clip
/// probed and planned under the context the request's text leaves
/// ([`clip_budget`]); on one that
/// takes none, its text parts joined, any other part refused by
/// name ([`content_text`]); a content that is neither a string, an array nor
/// `null` is refused too, as is a `tools` that is not an array. Nothing is
/// dropped. A `developer` message reaches the template as `system`, as
/// llama-server maps it, a tool call's string `arguments` reach a template
/// that reads them as an object parsed ([`tool_call_arguments`]), and the
/// thinking vars ([`thinking_vars`]) reach it in llama-server's order. A
/// special-token spelling in the client's text (a message, a tool call, a tool,
/// a `chat_template_kwargs` value; not an assistant message's content or
/// reasoning, which the model wrote) reaches the template marked
/// ([`crate::guard`]) and is text in the prompt.
fn render_chat(state: &State, b: &Map<String, Value>) -> Result<Rendered, ApiError> {
    let Some(Value::Array(msgs)) = b.get("messages") else {
        return Err(invalid("'messages' is required and must be an array").at("messages"));
    };
    // The media part refuses a typed placeholder by name; the guard leaves such a string to it.
    let markers: Vec<&str> = state
        .media
        .iter()
        .flat_map(|m| std::iter::once(m.image_marker()).chain(m.video_marker()))
        .collect();
    let guard = Guard::new(
        &*state.tok,
        b,
        &[
            state.template.source(),
            &state.info.bos_text,
            &state.info.eos_text,
        ],
    )
    .except(&markers);
    // GPT-OSS's template is the one that renders the `developer` role itself,
    // named by the `<|channel|>` marker in its source.
    let gpt_oss = state.template.source().contains("<|channel|>");
    let mut messages = Vec::with_capacity(msgs.len());
    let mut media: Vec<Medium> = Vec::new();
    let mut video = VideoCtx::new(
        state.video.clone(),
        if carries_video(msgs) {
            clip_budget(state, msgs)
        } else {
            0
        },
    );
    for (i, m) in msgs.iter().enumerate() {
        let Value::Object(m) = m else {
            return Err(invalid("each message must be an object").at("messages"));
        };
        let Some(role) = m.get("role").and_then(Value::as_str) else {
            return Err(invalid("each message needs a string 'role'").at("messages"));
        };
        // llama-server maps `developer` to `system` before the template: a
        // template without a branch for the role drops the message, and
        // OpenAI clients send one every turn. The mapped role is what the
        // content flattening and the template both see, as there the mapping
        // also runs before anything reads a role.
        let role = if role == "developer" && !gpt_oss {
            "system"
        } else {
            role
        };
        let mut flat = m.clone();
        flat.insert("role".into(), Value::String(role.to_owned()));
        // An assistant message's content is the model's own output, sent back as it came: its
        // special tokens' spellings are their tokens.
        let client = role != "assistant";
        match m.get("content") {
            Some(Value::Array(parts)) => {
                let text = match &state.media {
                    None => guard
                        .text_if(client, &content_text(parts).map_err(|e| e.at("messages"))?)
                        .into_owned(),
                    Some(model) => {
                        let first = media
                            .iter()
                            .filter(|m| matches!(m, Medium::Image(_)))
                            .count();
                        let f =
                            media::flatten_with(role, parts, &**model, first, &mut video, &|t| {
                                guard.text_if(client, t)
                            })
                            .map_err(|e| media_error(&e, "messages"))?;
                        media.extend(f.media);
                        f.text
                    }
                };
                flat.insert("content".into(), Value::String(text));
            }
            Some(Value::String(text)) => {
                if let Cow::Owned(g) = guard.text_if(client, text) {
                    flat.insert("content".into(), Value::String(g));
                }
            }
            None | Some(Value::Null) => {}
            Some(other) => {
                return Err(wrong_type(
                    &format!("messages[{i}].content"),
                    "a string or an array of parts",
                    other,
                ));
            }
        }
        if state.object_arguments {
            tool_call_arguments(i, &mut flat)?;
        }
        guard.message(role, &mut flat);
        messages.push(Value::Object(flat));
    }
    let mut vars = Map::new();
    vars.insert(
        "add_generation_prompt".into(),
        Value::Bool(get_b(b, "add_generation_prompt")?.unwrap_or(true)),
    );
    vars.insert(
        "bos_token".into(),
        Value::String(state.info.bos_text.clone()),
    );
    vars.insert(
        "eos_token".into(),
        Value::String(state.info.eos_text.clone()),
    );
    match b.get("tools") {
        None | Some(Value::Null) => {}
        Some(v @ Value::Array(_)) => {
            let mut tools = v.clone();
            guard.value(&mut tools);
            vars.insert("tools".into(), tools);
        }
        Some(other) => return Err(wrong_type("tools", "an array", other)),
    }
    thinking_vars(b, &mut vars, state.thinking_off)?;
    if let Some(preparation) = state.preparation {
        // The family's preparation shapes the request before the template
        // renders it: it rewrites the messages and adds the variables its
        // template reads.
        let mut request = ChatRequest {
            messages: &mut messages,
            body: b,
            vars: &mut vars,
            marker: &|value| guard.value(value),
        };
        preparation(&mut request).map_err(|e| invalid(e).at("messages"))?;
    }
    // The messages go in last: a preparation may have rewritten them.
    vars.insert("messages".into(), Value::Array(messages));
    if let Some(Value::Object(kwargs)) = b.get("chat_template_kwargs") {
        // What the kwargs put into the vars as they came is the client's text.
        for (k, v) in kwargs {
            if let Some(var) = vars.get_mut(k)
                && var == v
            {
                guard.value(var);
            }
        }
    }
    let marked = state.template.render(&vars).map_err(template_error)?;
    let (text, guarded) = guard
        .finish(marked, &state.template, &vars)
        .map_err(guard_error)?;
    Ok(Rendered {
        text,
        guarded,
        media,
    })
}

/// A template that refused the request (`raise_exception`) names the client's
/// request, which is a 400; any other render error is the template's own,
/// which is a 500.
fn template_error(e: TemplateError) -> ApiError {
    match e.refusal() {
        Some(refusal) => invalid(refusal.to_owned()),
        None => ApiError {
            retry_after: false,
            param: None,
            code: 500,
            kind: "server_error",
            message: e.to_string(),
        },
    }
}

/// A template that cut a guarded text apart is the request's 400 naming it.
fn guard_error(e: GuardError) -> ApiError {
    invalid(e.to_string())
}

/// Whether a message of `msgs` carries a clip part.
fn carries_video(msgs: &[Value]) -> bool {
    msgs.iter()
        .filter_map(|m| m.get("content")?.as_array())
        .flatten()
        .any(|p| {
            matches!(
                p.get("type").and_then(Value::as_str),
                Some("input_video" | "video_url")
            )
        })
}

/// The positions of context the request's clips may take together: the engine's
/// context less the tokens of the request's text parts and the template's own
/// ([`video::clip_budget`]). The text is counted by the model's tokenizer, so a
/// long prompt leaves a short clip its due.
fn clip_budget(state: &State, msgs: &[Value]) -> usize {
    let text_tokens: usize = msgs
        .iter()
        .map(|m| match m.get("content") {
            Some(Value::String(s)) => state.tok.encode(s).len(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|p| p.get("text")?.as_str())
                .map(|t| state.tok.encode(t).len())
                .sum(),
            _ => 0,
        })
        .sum();
    video::clip_budget(state.info.ctx_max, text_tokens)
}

/// The argument a template's probe call carries.
const PROBE_ARGUMENT: &str = "argument_needle";

/// Whether the chat template needs a tool call's `arguments` as an object
/// rather than the JSON string the OpenAI API carries: a probe call renders
/// with its arguments as an object and as that object's JSON string, and the
/// template needs the object when the object's render shows the argument and
/// the string's does not — it fails on the string (`.items()`, a
/// `raise_exception`), drops it, or prints it as a quoted JSON string. That is
/// minja's `requires_object_arguments`; llama-server now decides by what the
/// template touches of the object (`supports_object_arguments`,
/// `common/jinja/caps.cpp`), which this engine cannot observe, and parses as
/// well for a template that takes a string and an object alike but renders an
/// object through `tojson`, which this server passes as it came.
fn needs_object_arguments(t: &ChatTemplate) -> bool {
    let object = json!({ PROBE_ARGUMENT: "print('Hello, World!')" });
    let render = |arguments: Value| {
        let mut vars = Map::new();
        vars.insert(
            "messages".into(),
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {
                        "name": "probe", "arguments": arguments,
                    }},
                ]},
            ]),
        );
        vars.insert("add_generation_prompt".into(), json!(false));
        vars.insert("bos_token".into(), json!(""));
        vars.insert("eos_token".into(), json!(""));
        vars.insert(
            "tools".into(),
            json!([{"type": "function", "function": {
                "name": "probe", "parameters": {"type": "object"},
            }}]),
        );
        t.render(&vars).unwrap_or_default()
    };
    let escaped = format!("\\\"{PROBE_ARGUMENT}");
    let as_string = render(Value::String(object.to_string()));
    render(object).contains(PROBE_ARGUMENT)
        && !(as_string.contains(PROBE_ARGUMENT) && !as_string.contains(&escaped))
}

/// How a chat template answers a request for no thinking
/// (`reasoning_effort: "none"`), read from its renders once at bind
/// ([`probe_thinking_off`]), never from its name or the model's family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThinkingOff {
    /// `enable_thinking` changes the generation prompt: `"none"` sets it
    /// `false`, as llama-server does. With `opt_in` the template renders no
    /// thinking until it is `true` (V4.1's), so any other effort sets it
    /// `true` unless the request set it.
    Switch { opt_in: bool },
    /// The template has no switch but `reasoning_effort` changes the
    /// generation prompt: `"none"` is sent as the template's lowest effort,
    /// `"low"`, the closest to no thinking it has.
    LowEffort,
    /// Neither knob changes the prompt: `"none"` sets `enable_thinking`
    /// `false`, as llama-server does, and there is nothing to map it to.
    Neither,
}

impl ThinkingOff {
    /// The word `/props`' `thinking_off` carries.
    fn word(self) -> &'static str {
        match self {
            ThinkingOff::Switch { .. } => "switch",
            ThinkingOff::LowEffort => "low_effort",
            ThinkingOff::Neither => "neither",
        }
    }

    /// The line the server prints at bind when `"none"` is not what the
    /// client asked for; `None` for the two ways that keep llama-server's
    /// meaning.
    fn note(self) -> Option<&'static str> {
        match self {
            ThinkingOff::LowEffort => Some(
                "chat template: no thinking switch; reasoning_effort \"none\" is sent as \"low\"",
            ),
            ThinkingOff::Switch { .. } | ThinkingOff::Neither => None,
        }
    }
}

/// Reads the template's [`ThinkingOff`] from the generation prompt of one
/// user message `"hi"`: a template where `enable_thinking: false` and `true`
/// render differently has a [`Switch`](ThinkingOff::Switch), opt-in when
/// the prompt with no `enable_thinking` is not the `true` one; otherwise one
/// where `reasoning_effort: "low"` renders differently from no effort is
/// [`LowEffort`](ThinkingOff::LowEffort); otherwise
/// [`Neither`](ThinkingOff::Neither). A render that fails is the bind's
/// error, naming the probe: the answer is never guessed.
fn probe_thinking_off(t: &ChatTemplate) -> Result<ThinkingOff, TemplateError> {
    let render = |extra: Option<(&str, Value)>| {
        let mut vars = Map::new();
        vars.insert(
            "messages".into(),
            json!([{"role": "user", "content": "hi"}]),
        );
        vars.insert("add_generation_prompt".into(), json!(true));
        vars.insert("bos_token".into(), json!(""));
        vars.insert("eos_token".into(), json!(""));
        if let Some((k, v)) = extra {
            vars.insert(k.into(), v);
        }
        t.render(&vars)
            .map_err(|e| TemplateError(format!("thinking probe: {}", e.0)))
    };
    let off = render(Some(("enable_thinking", json!(false))))?;
    let on = render(Some(("enable_thinking", json!(true))))?;
    if off != on {
        let opt_in = render(None)? != on;
        return Ok(ThinkingOff::Switch { opt_in });
    }
    let unset = render(None)?;
    let low = render(Some(("reasoning_effort", json!("low"))))?;
    Ok(if low != unset {
        ThinkingOff::LowEffort
    } else {
        ThinkingOff::Neither
    })
}

/// The tool calls of message `at` with each string `arguments` parsed as JSON,
/// as llama-server's template application does for a template that reads
/// objects (`func_args_not_string`, `common/chat.cpp`); a string that is not
/// JSON is a 400 naming the message, the call and the field.
fn tool_call_arguments(at: usize, message: &mut Map<String, Value>) -> Result<(), ApiError> {
    let Some(Value::Array(calls)) = message.get_mut("tool_calls") else {
        return Ok(());
    };
    for (j, call) in calls.iter_mut().enumerate() {
        let Some(Value::String(text)) = call.get("function").and_then(|f| f.get("arguments"))
        else {
            continue;
        };
        let parsed = serde_json::from_str::<Value>(text).map_err(|e| {
            invalid(format!(
                "messages[{at}].tool_calls[{j}].function.arguments is not valid JSON: {e}"
            ))
            .at("messages")
        })?;
        call["function"]["arguments"] = parsed;
    }
    Ok(())
}

/// The values OpenAI's `reasoning_effort` takes (the `ReasoningEffort` type of
/// its SDK), and so the Responses API's `reasoning.effort`. A template takes
/// the ones it knows; `max` is V4.1's maximum-effort header and `xhigh`,
/// `minimal` and `none` are OpenAI's own words.
const REASONING_EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// An effort that is none of [`REASONING_EFFORTS`] is a 400 naming the field,
/// the value and the accepted ones: a template would render a word it does not
/// know as itself, or not at all, and the request would run as if it had named
/// none.
fn check_effort(field: &str, effort: &str) -> Result<(), ApiError> {
    if REASONING_EFFORTS.contains(&effort) {
        return Ok(());
    }
    Err(invalid(format!(
        "invalid value for \"{field}\": {effort:?} (expected one of {})",
        REASONING_EFFORTS.join(", ")
    ))
    .at(field))
}

/// The thinking vars a chat body carries, in llama-server's order: the kwargs
/// (`chat_template_kwargs`, an object) `enable_thinking` first — a bool sets
/// it, a string is a 400 naming it —
/// then the OpenAI `reasoning_effort`, whose `"none"` turns thinking off and
/// does not reach the template as itself (a kwargs `reasoning_effort` is
/// erased with it). The resolved `enable_thinking` replaces the kwargs entry,
/// as llama-server's template application writes the resolved value over it,
/// so `"none"` wins over a kwargs `true`. It lands in `vars` only when
/// something asked: a template that tells an undefined `enable_thinking` from
/// `true` (V4.1's takes its `thinking` from it) keeps its own default when
/// nothing did.
///
/// Any other effort asks for reasoning: it reaches the template as itself,
/// and a template that renders no thinking until `enable_thinking` is `true`
/// ([`ThinkingOff::Switch`] with `opt_in`) gets it `true` from the effort
/// unless the kwargs set it, which wins either way. (llama-server's own
/// default is thinking on for a template that has the switch; this server
/// keeps the template's default, so the effort is what turns it on.)
///
/// A template with no thinking switch but an effort knob
/// ([`ThinkingOff::LowEffort`]) has nothing `enable_thinking: false` could
/// turn off, so there `"none"` is the effort `"low"`: it sets no
/// `enable_thinking` of its own and replaces a kwargs `reasoning_effort`.
fn thinking_vars(
    b: &Map<String, Value>,
    vars: &mut Map<String, Value>,
    how: ThinkingOff,
) -> Result<(), ApiError> {
    let kwargs = match b.get("chat_template_kwargs") {
        None | Some(Value::Null) => None,
        Some(Value::Object(kw)) => Some(kw),
        Some(other) => return Err(wrong_type("chat_template_kwargs", "an object", other)),
    };
    let mut thinking: Option<bool> = None;
    if let Some(kw) = kwargs {
        match kw.get("enable_thinking") {
            Some(Value::Bool(on)) => thinking = Some(*on),
            Some(Value::String(_)) => {
                return Err(invalid(
                    "invalid type for \"enable_thinking\" (expected boolean, got string)",
                )
                .at("chat_template_kwargs"));
            }
            _ => {}
        }
    }
    let effort = match b.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(v @ Value::String(e)) => {
            check_effort("reasoning_effort", e)?;
            Some(v)
        }
        Some(other) => return Err(wrong_type("reasoning_effort", "a string", other)),
    };
    let none = effort.and_then(Value::as_str) == Some("none");
    match (none, how) {
        (true, ThinkingOff::LowEffort) => {
            vars.insert("reasoning_effort".into(), Value::String("low".to_owned()));
        }
        (true, ThinkingOff::Switch { .. } | ThinkingOff::Neither) => thinking = Some(false),
        (false, how) => {
            if let Some(v) = effort {
                vars.insert("reasoning_effort".into(), v.clone());
                if thinking.is_none() && how == (ThinkingOff::Switch { opt_in: true }) {
                    thinking = Some(true);
                }
            }
        }
    }
    if let Some(on) = thinking {
        vars.insert("enable_thinking".into(), Value::Bool(on));
    }
    if let Some(kw) = kwargs {
        // The resolved bool replaces a kwargs `enable_thinking`; any other
        // value of it passes through as it came, as llama-server's resolution
        // leaves it alone.
        vars.extend(
            kw.iter()
                .filter(|(k, _)| {
                    !(thinking.is_some() && k.as_str() == "enable_thinking")
                        && !(none && k.as_str() == "reasoning_effort")
                })
                .map(|(k, v)| (k.clone(), v.clone())),
        );
    }
    Ok(())
}

/// The text of an array `content` on an engine that takes no images: its text
/// parts joined by newlines. A media part is a 400 naming its kind in
/// llama-server's words, and any other part a 400 too: nothing is dropped.
fn content_text(parts: &[Value]) -> Result<String, ApiError> {
    let no_media = |kind: &str| {
        invalid(format!(
            "{kind} input is not supported: this server reads text only"
        ))
    };
    let mut text = Vec::with_capacity(parts.len());
    for p in parts {
        match p.get("type").and_then(Value::as_str) {
            Some("text") => match p.get("text") {
                Some(Value::String(s)) => text.push(s.as_str()),
                _ => return Err(invalid("content[].text must be a string")),
            },
            Some("image_url") => return Err(no_media("image")),
            Some("input_audio") => return Err(no_media("audio")),
            Some("input_video" | "video_url") => return Err(no_media("video")),
            _ => return Err(invalid("unsupported content[].type")),
        }
    }
    Ok(text.join("\n"))
}

// ---------------------------------------------------------------- generation endpoints

fn dead_engine(reason: &str) -> ApiError {
    ApiError {
        retry_after: false,
        param: None,
        code: 503,
        kind: "unavailable_error",
        message: stopping(reason),
    }
}

/// The 503 a request that needs the engine gets once the server's orderly
/// stop began: it names the stop's cause and asks for no retry — the process
/// is ending.
fn stopping_gate(cause: &str) -> ApiError {
    ApiError {
        retry_after: false,
        param: None,
        code: 503,
        kind: "unavailable_error",
        message: stopping_message(cause),
    }
}

/// The answer to a request or an action the board did not take.
fn refused(state: &State, r: Refusal, what: &str) -> ApiError {
    match r {
        Refusal::Full { depth } => ApiError {
            retry_after: true,
            param: None,
            code: 503,
            kind: "unavailable_error",
            message: format!(
                "every one of the {} slots is busy and {depth} requests wait, the queue's depth",
                state.parallel
            ),
        },
        Refusal::Busy => ApiError {
            retry_after: false,
            param: None,
            code: 503,
            kind: "unavailable_error",
            message: format!("{what} is processing a request"),
        },
        // The board dies with the stop's cause too: a request that passed
        // the stop's check before the halt killed the board names the stop.
        Refusal::Dead(reason) => match state.stopping() {
            Some(cause) => stopping_gate(&cause),
            None => dead_engine(&reason),
        },
    }
}

/// What a request's HTTP thread hears when the engine thread closed its
/// channel without a last message: the orderly stop's 503, when the stop
/// closed it; else the failure that ended it, a 500 for a request that had
/// started, a 503 otherwise.
fn engine_gone(state: &State, started: bool) -> ApiError {
    if let Some(cause) = state.stopping() {
        return stopping_gate(&cause);
    }
    let reason = state
        .fatal()
        .unwrap_or_else(|| "the engine thread ended".to_owned());
    if started {
        engine_error(&reason)
    } else {
        dead_engine(&reason)
    }
}

/// A refusal of the media part: the request's 400. An empty span is the model's
/// 500, because message text that carries the marker is refused before the
/// prompt is tokenized; the clip's temp file that cannot be written is the
/// server's 500. A numbered placeholder whose ids are not its image's
/// position is the request's 400, with the placeholder-count and image-last faults.
fn media_error(e: &MediaError, field: &str) -> ApiError {
    match e {
        MediaError::Expand(ExpandError::EmptySpan(_))
        | MediaError::Clip(video::VideoError::Temp(_)) => engine_error(e),
        _ => invalid(e.to_string()).at(field),
    }
}

/// The prompt of a rendered chat: its ids, and on an engine that takes images
/// each image expanded to its span with its feed ([`media::expand_prompt_with`]
/// over the server's prepared images), which also refuses a placeholder no
/// image part made.
fn chat_prompt(state: &State, rendered: &Rendered) -> Result<Prompt, ApiError> {
    let ids = rendered.ids(&*state.tok);
    let Some(model) = &state.media else {
        return Ok(Prompt::from(ids));
    };
    media::expand_prompt_with(&ids, &rendered.media, &**model, &state.images)
        .map_err(|e| media_error(&e, "messages"))
}

/// What the 400 for a prompt past the context adds to its two numbers: the
/// one flag that serves it, `--ctx-size` of the smallest power of two that
/// holds the prompt and what the request asks to generate (`n_predict`, when
/// it is positive; one token when it sets none or `0`, the least a prompt that
/// fits can still produce). With more than one slot the flag is also given for
/// a seat whose slots split it under `--parallel` (the others give each slot
/// the whole flag, and the clause says where it applies). Empty when no power
/// of two fits a `usize`.
fn ctx_hint(prompt_tokens: usize, n_predict: i64, slots: usize) -> String {
    let generated = usize::try_from(n_predict)
        .ok()
        .filter(|&n| n > 0)
        .unwrap_or(1);
    let Some(size) = prompt_tokens
        .saturating_add(generated)
        .checked_next_power_of_two()
    else {
        return String::new();
    };
    let mut hint = format!("; serve it with --ctx-size {size}");
    if slots > 1 {
        hint.push_str(&format!(
            " (under --parallel {slots}, where the slots split the flag: --ctx-size {})",
            size.saturating_mul(slots)
        ));
    }
    hint
}

/// [`run_choices`] of a request of one choice: its outcome and the slot it
/// took.
fn run_gen(
    state: &State,
    route: &str,
    input: &Prompt,
    prompt: Value,
    p: &GenParams,
    client: Arc<Client>,
    sink: &mut dyn FnMut(Event<'_>, usize) -> io::Result<()>,
) -> Result<Result<(Outcome, usize), GenError>, ApiError> {
    let ran = run_choices(
        state,
        route,
        input,
        prompt,
        std::slice::from_ref(p),
        client,
        &mut |ev, _, slot| sink(ev, slot),
    )?;
    Ok(ran.map(|mut all| all.swap_remove(0)))
}

/// Validates the prompt and builds each choice's sampler (a factory's
/// refusal is a 400), hands the choices to the engine thread together (the
/// board queues all of them or refuses the request, [`Board::admits`]), and
/// passes their events to `sink` with the choice (its index in `ps`, one
/// `GenParams` a choice) and the slot it took. `input` is what the engine is
/// fed: the ids, and the images' spans and feeds of a chat that carries any.
/// Returns every choice's outcome and slot, in the choices' order, once each
/// one is done; an error before any event is the request's answer. The first
/// choice that fails, a sink that fails, and a `client` gone end the
/// request: the choices still queued leave the queue, and the running ones
/// end at their next event, the channel they send on closed.
fn run_choices(
    state: &State,
    route: &str,
    input: &Prompt,
    prompt: Value,
    ps: &[GenParams],
    client: Arc<Client>,
    sink: &mut dyn FnMut(Event<'_>, usize, usize) -> io::Result<()>,
) -> Result<Result<Vec<(Outcome, usize)>, GenError>, ApiError> {
    if let Some(reason) = state.fatal() {
        return Err(dead_engine(&reason));
    }
    if let Some(cause) = state.stopping() {
        return Err(stopping_gate(&cause));
    }
    let ids = &input.held.ids;
    if ids.is_empty() {
        return Err(invalid("the prompt is empty"));
    }
    let n_predict = ps.first().map_or(-1, |p| p.n_predict);
    if ids.len() >= state.info.ctx_max {
        return Err(ApiError {
            retry_after: false,
            param: None,
            code: 400,
            kind: "exceed_context_size_error",
            message: format!(
                "the prompt has {} tokens and the context holds {}{}",
                ids.len(),
                state.info.ctx_max,
                ctx_hint(ids.len(), n_predict, state.parallel)
            ),
        });
    }
    let (events, rx) = mpsc::channel();
    // A sampler its factory refuses ends the request here, before it takes a
    // slot, as any refused sampling field does.
    let subs = ps
        .iter()
        .enumerate()
        .map(|(choice, p)| {
            let sampler = genloop::sampler_of(&state.shared.sampler, p).map_err(|e| {
                invalid(format!("the sampler refuses the sampling parameters: {e}"))
            })?;
            Ok(Submit {
                p: p.clone(),
                prompt: prompt.clone(),
                settings: generation_settings(state, p),
                media: input.feeds.clone(),
                sampler,
                events: Events::new(events.clone(), choice, client.place.upgrade()),
                client: Arc::clone(&client),
                name: ReqName {
                    id: state.next_id(),
                    route: route.to_owned(),
                    choice,
                    choices: ps.len(),
                },
            })
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    drop(events);
    {
        let mut board = relock(&state.shared.board);
        board
            .admits(subs.len())
            .map_err(|r| refused(state, r, "the slot"))?;
        for sub in subs {
            board
                .enqueue(input.held.clone(), sub)
                .map_err(|r| refused(state, r, "the slot"))?;
        }
    }
    state.shared.work.notify_one();
    // The request ends before every choice took a slot: the ones still
    // queued leave it, each counted as cancelled before its prompt.
    let leave = || {
        let mut board = relock(&state.shared.board);
        let mut left = 0;
        while board.cancel(|sub| Arc::ptr_eq(&sub.client, &client)) {
            left += 1;
        }
        drop(board);
        if left > 0 {
            relock(&state.shared.stats).n_cancelled_before_prompt_total += left;
        }
    };
    let stream = ps.first().is_some_and(|p| p.stream);
    let mut slots: Vec<Option<usize>> = vec![None; ps.len()];
    let mut done: Vec<Option<Outcome>> = ps.iter().map(|_| None).collect();
    loop {
        let started = slots.iter().all(Option::is_some);
        if (!started || !stream)
            && let Err(e) = client.check()
        {
            leave();
            return Ok(Err(GenError::Client(e)));
        }
        let received = if stream && started {
            rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected)
        } else {
            rx.recv_timeout(Duration::from_millis(100))
        };
        let (choice, m) = match received {
            Ok(m) => m,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(engine_gone(state, slots.iter().any(Option::is_some)));
            }
        };
        let at = slots[choice].unwrap_or(0);
        let sent = match m {
            Msg::Started(s) => {
                slots[choice] = Some(s);
                Ok(())
            }
            Msg::Prompt(t) => sink(Event::Prompt(&t), choice, at),
            Msg::Text(text, specials, t) => sink(Event::Text(&text, &specials, &t), choice, at),
            Msg::Done(Ok(o)) => {
                done[choice] = Some(o);
                if done.iter().all(Option::is_some) {
                    let slots = slots.iter().map(|s| s.unwrap_or(0));
                    return Ok(Ok(done.into_iter().flatten().zip(slots).collect()));
                }
                Ok(())
            }
            Msg::Done(Err(e)) => {
                leave();
                return Ok(Err(e));
            }
            Msg::Refused(why) => {
                leave();
                return Err(ApiError {
                    retry_after: true,
                    param: None,
                    code: 503,
                    kind: "unavailable_error",
                    message: why,
                });
            }
        };
        if let Err(e) = sent {
            leave();
            return Ok(Err(GenError::Client(e)));
        }
    }
}

/// What a request's generation fails with once its client is gone: the sink's
/// error, the one the engine thread's channel send fails with as well.
pub(crate) fn client_gone() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "the client went away")
}

/// The response side's failure shared by the HTTP and engine threads. A socket
/// error is consumed by `take_error`, so the first observer remembers it.
pub(crate) struct Client {
    /// The request's connection; `None` for a background response, which no
    /// connection waits on.
    stream: Option<TcpStream>,
    gone: Mutex<bool>,
    /// The request's key place: each of its engine jobs holds a share until
    /// it ends ([`Events`]). Weak, so the client itself holds none.
    place: Weak<admit::Place>,
}

impl Client {
    /// The client of `req` on the request's connection `stream`.
    fn of(stream: TcpStream, req: &Request) -> Arc<Client> {
        Arc::new(Client {
            stream: Some(stream),
            place: Client::place(req),
            ..Client::detached()
        })
    }

    /// A client with no connection, which only [`Client::cancel`] ends.
    fn detached() -> Client {
        Client {
            stream: None,
            gone: Mutex::new(false),
            place: Weak::new(),
        }
    }

    /// `req`'s place, weak.
    fn place(req: &Request) -> Weak<admit::Place> {
        relock(&req.lease)
            .as_ref()
            .map_or_else(Weak::new, Arc::downgrade)
    }

    /// Ends the request at its next check, as a client that went away does.
    fn cancel(&self) {
        *relock(&self.gone) = true;
    }

    pub(crate) fn check(&self) -> io::Result<()> {
        let mut gone = relock(&self.gone);
        if *gone {
            return Err(client_gone());
        }
        let Some(stream) = &self.stream else {
            return Ok(());
        };
        let result = probe_client(stream);
        #[cfg(test)]
        if result.is_err() {
            testserve::CLIENT_PROBE.with_borrow_mut(|pause| {
                if let Some((probed, resume)) = pause.take() {
                    probed.send(()).expect("the probe observer");
                    resume
                        .recv_timeout(Duration::from_secs(10))
                        .expect("the probe resumes");
                }
            });
        }
        *gone = result.is_err();
        result
    }
}

/// A transport error ends a request. Read EOF alone also describes a client
/// that closed only its write side and is still waiting for the response.
fn probe_client(w: &TcpStream) -> io::Result<()> {
    match w.take_error()? {
        None => Ok(()),
        Some(e) => match e.kind() {
            io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
            | io::ErrorKind::NotConnected => Err(client_gone()),
            _ => Err(e),
        },
    }
}

/// [`run_whole_choices`] of a request of one choice.
fn run_whole(
    state: &State,
    input: &Prompt,
    prompt: Value,
    p: &GenParams,
    req: &Request,
    w: &TcpStream,
) -> io::Result<Result<(Outcome, usize), ApiError>> {
    let ran = run_whole_choices(state, input, prompt, std::slice::from_ref(p), req, w)?;
    Ok(ran.map(|mut all| all.swap_remove(0)))
}

/// [`run_choices`] for a request that answers once every generation is
/// whole: it sends nothing while they run. The shared transport check ends a
/// failed connection while preserving a client's write-side shutdown. The
/// outer error is that client's, whose connection ends with no answer; the
/// inner one is the request's answer, an engine error a 500.
fn run_whole_choices(
    state: &State,
    input: &Prompt,
    prompt: Value,
    ps: &[GenParams],
    req: &Request,
    w: &TcpStream,
) -> io::Result<Result<Vec<(Outcome, usize)>, ApiError>> {
    match run_choices(
        state,
        &req.path,
        input,
        prompt,
        ps,
        Client::of(w.try_clone()?, req),
        &mut |_, _, _| Ok(()),
    ) {
        Err(e) => Ok(Err(e)),
        Ok(Err(GenError::Client(e))) => Err(e),
        Ok(Err(e)) => Ok(Err(engine_error(&e))),
        Ok(Ok(all)) => Ok(Ok(all)),
    }
}

// ---------------------------------------------------------------- slot actions

enum SlotAction {
    Save(String),
    Restore(String),
    Erase,
}

/// Runs `run` on the engine thread once the board reserves `what` (free, no
/// request waiting; else a 503 at once naming `who`), and answers with what it
/// returns.
fn on_engine(
    state: &State,
    what: Reserve,
    who: &str,
    drops: bool,
    run: impl FnOnce(&mut Slot) -> (Result<Value, ApiError>, Option<EngineError>) + Send + 'static,
) -> Result<Value, ApiError> {
    if let Some(reason) = state.fatal() {
        return Err(dead_engine(&reason));
    }
    if let Some(cause) = state.stopping() {
        return Err(stopping_gate(&cause));
    }
    let (tx, rx) = mpsc::channel();
    let action = Action {
        run: Box::new(move |slot: &mut Slot| {
            let (answer, failure) = run(slot);
            Acted {
                failure,
                reply: Box::new(move || {
                    let _ = tx.send(answer);
                }),
            }
        }),
        drops,
    };
    relock(&state.shared.board)
        .reserve(what, action)
        .map_err(|r| refused(state, r, who))?;
    state.shared.work.notify_one();
    rx.recv().unwrap_or_else(|_| Err(engine_gone(state, true)))
}

/// `POST /residency/reset`: the engine's adaptive expert residency back to
/// its load's placement ([`Engine::residency_reset`]), while every slot is
/// free and no request waits (else a 503 at once). An engine with no
/// residency is a 501; an engine that fails is fatal, as any engine error.
fn residency_reset(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let who = if state.parallel == 1 {
        "slot 0"
    } else {
        "a slot"
    };
    let r = on_engine(state, Reserve::All, who, false, |slot| {
        let t0 = Instant::now();
        match slot.engine.residency_reset() {
            Ok(Some(r)) => (
                Ok(json!({
                    "cancelled": r.cancelled,
                    "copies": r.copies,
                    "diff": r.diff,
                    "dropped_bytes": r.dropped_bytes,
                    "timings": { "reset_ms": ms_since(t0) },
                })),
                None,
            ),
            Ok(None) => (
                Err(ApiError {
                    retry_after: false,
                    param: None,
                    code: 501,
                    kind: "not_supported_error",
                    message: "this engine runs no adaptive expert residency".to_owned(),
                }),
                None,
            ),
            Err(e) => (Err(engine_error(&e)), Some(e)),
        }
    });
    match r {
        Ok(v) => send_json(w, req, 200, &v),
        Err(e) => send_error(w, req, &e),
    }
}

/// Whether `peer` may ask the server to stop over `POST /shutdown`: only a
/// loopback peer — IPv4 loopback, `::1`, or an IPv4-mapped loopback address —
/// so a request from another machine cannot stop a server bound wider than
/// its operator meant. The signals need no check: the OS delivers them to
/// this process.
fn shutdown_allowed(peer: IpAddr) -> bool {
    match peer {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => {
            ip.is_loopback() || matches!(ip.to_ipv4_mapped(), Some(v4) if v4.is_loopback())
        }
    }
}

/// `POST /shutdown`, bloomery's own: the same orderly stop the first
/// `SIGINT`/`SIGTERM` begins, asked over HTTP — accepted only from a loopback
/// peer ([`shutdown_allowed`]), else a 403 saying so. The 200 reaches the
/// client before the stop does: the answer is sent first, then the end. A
/// `POST /shutdown` while a stop is already under way answers 200 again and
/// changes nothing.
fn post_shutdown(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let peer = match w.peer_addr() {
        Ok(a) => a,
        Err(e) => {
            return send_error(
                w,
                req,
                &ApiError {
                    retry_after: false,
                    param: None,
                    code: 500,
                    kind: "server_error",
                    message: format!("cannot read the connection's peer: {e}"),
                },
            );
        }
    };
    if !shutdown_allowed(peer.ip()) {
        return send_error(
            w,
            req,
            &ApiError {
                retry_after: false,
                param: None,
                code: 403,
                kind: "permission_error",
                message: "shutdown is accepted only from this machine (a loopback peer)".to_owned(),
            },
        );
    }
    let stop = Stop::Posted(peer);
    let began = state.shared.begin_stop(&stop);
    send_json(w, req, 200, &json!({ "status": "shutting down" }))?;
    if began {
        let _ = state.shared.end.send(End::Shutdown(stop));
    }
    Ok(true)
}

/// `POST /slots/{id}?action=…`, checked in llama-server's order: a save
/// directory, an integer id, the action, the file name, then that the id is
/// one of the slots'; last the slot itself, which must be free.
fn slot_action(state: &State, req: &Request, w: &mut TcpStream, id: &str) -> io::Result<bool> {
    let Some(dir) = state.slot_save_path.clone() else {
        return send_error(
            w,
            req,
            &ApiError {
                retry_after: false,
                param: None,
                code: 501,
                kind: "not_supported_error",
                message: "This server does not support slots action. Start it with \
                          `--slot-save-path`"
                    .to_owned(),
            },
        );
    };
    let (id, action) = match slot_request(req, id, state.parallel) {
        Ok(a) => a,
        Err(e) => return send_error(w, req, &e),
    };
    let drops = matches!(action, SlotAction::Erase);
    let r = on_engine(
        state,
        Reserve::One(id),
        &format!("slot {id}"),
        drops,
        move |slot| slot_job(slot, id, &dir, action),
    );
    match r {
        Ok(v) => send_json(w, req, 200, &v),
        Err(e) => send_error(w, req, &e),
    }
}

/// The slot id and the action with its file name; every refusal is a 400.
fn slot_request(req: &Request, id: &str, slots: usize) -> Result<(usize, SlotAction), ApiError> {
    let id: i64 = id.parse().map_err(|_| invalid("Invalid slot ID"))?;
    let action = req.query_value("action").unwrap_or_default();
    let filename = || -> Result<String, ApiError> {
        let b = body(req)?;
        let f = b
            .get("filename")
            .ok_or_else(|| invalid("'filename' is required").at("filename"))?
            .as_str()
            .ok_or_else(|| invalid("'filename' must be a string").at("filename"))?;
        if !slotfile::valid_filename(f) {
            return Err(invalid("Invalid filename").at("filename"));
        }
        Ok(f.to_owned())
    };
    let a = match action {
        "save" => SlotAction::Save(filename()?),
        "restore" => SlotAction::Restore(filename()?),
        "erase" => SlotAction::Erase,
        _ => return Err(invalid("Invalid action")),
    };
    match usize::try_from(id) {
        Ok(i) if i < slots => Ok((i, a)),
        _ => Err(invalid(format!(
            "Invalid slot ID {id}: the server runs slots 0 to {}",
            slots - 1
        ))),
    }
}

/// Runs a slot action on the selected slot and answers with llama-server's
/// object for it, and the engine error it met, which is fatal.
fn slot_job(
    slot: &mut Slot,
    id: usize,
    dir: &Path,
    action: SlotAction,
) -> (Result<Value, ApiError>, Option<EngineError>) {
    let t0 = Instant::now();
    match action {
        SlotAction::Erase => match slot.erase() {
            Ok(n) => (Ok(json!({ "id_slot": id, "n_erased": n })), None),
            Err(e) => (Err(engine_error(&e)), Some(e)),
        },
        SlotAction::Save(f) => match slot.save(&dir.join(&f)) {
            Ok((n, bytes)) => (
                Ok(json!({
                    "id_slot": id,
                    "filename": f,
                    "n_saved": n,
                    "n_written": bytes,
                    "timings": { "save_ms": ms_since(t0) },
                })),
                None,
            ),
            Err(e) => state_error(e, "Unable to save slot", 500, "server_error"),
        },
        SlotAction::Restore(f) => match slot.restore(&dir.join(&f)) {
            Ok((n, bytes)) => (
                Ok(json!({
                    "id_slot": id,
                    "filename": f,
                    "n_restored": n,
                    "n_read": bytes,
                    "timings": { "restore_ms": ms_since(t0) },
                })),
                None,
            ),
            Err(e) => state_error(e, "Unable to restore slot", 400, "invalid_request_error"),
        },
    }
}

/// A refused engine is a 501, a failed one fatal (500), anything else `code`
/// with `what` before the reason.
fn state_error(
    e: StateError,
    what: &str,
    code: u16,
    kind: &'static str,
) -> (Result<Value, ApiError>, Option<EngineError>) {
    match e {
        StateError::Unsupported(_) => (
            Err(ApiError {
                retry_after: false,
                param: None,
                code: 501,
                kind: "not_supported_error",
                message: e.to_string(),
            }),
            None,
        ),
        StateError::Engine(e) => (Err(engine_error(&e)), Some(e)),
        StateError::Format(_) | StateError::Io(_) | StateError::NoRoom { .. } => (
            Err(ApiError {
                retry_after: false,
                param: None,
                code,
                kind,
                message: format!("{what}: {e}"),
            }),
            None,
        ),
    }
}

/// The prompt of `/completion`: text (BOS per `add_bos_token`), an id array, or a
/// mixed array whose strings are tokenized in place. A prompt object carrying
/// media, alone or in the array, is refused by that name in llama-server's words;
/// on an engine that takes images, so is the image token, which only a chat's
/// image part makes.
fn completion_prompt(state: &State, v: Option<&Value>) -> Result<Vec<u32>, ApiError> {
    let ids = completion_ids(state, v)?;
    if ids.is_empty() {
        return Err(invalid("the prompt is empty").at("prompt"));
    }
    if let Some(model) = &state.media {
        media::expand_prompt(&ids, &[], &**model).map_err(|e| media_error(&e, "prompt"))?;
    }
    Ok(ids)
}

fn completion_ids(state: &State, v: Option<&Value>) -> Result<Vec<u32>, ApiError> {
    match v {
        Some(o) if carries_media(o) => Err(no_media().at("prompt")),
        Some(v @ (Value::String(_) | Value::Array(_))) => {
            prompt_ids(&*state.tok, "prompt", v, true)
        }
        _ => Err(invalid("'prompt' must be a string or an array of tokens").at("prompt")),
    }
}

fn no_media() -> ApiError {
    ::embed::input::no_media().into()
}

/// The field `k`, a string or an array of strings and token ids, as `tok`'s token ids, as
/// llama-server's `tokenize_mixed` reads it: each string encoded, each id checked against the
/// vocabulary. `special` gives the input's first element, when it is text, the vocabulary's
/// special tokens: its BOS first ([`Tokenizer::add_bos`]) and its EOS after it
/// ([`Tokenizer::add_eos`]). Any other type is a 400 naming `k` and the type it got.
fn prompt_ids(
    tok: &dyn Tokenizer,
    k: &str,
    v: &Value,
    special: bool,
) -> Result<Vec<u32>, ApiError> {
    let bos = if special {
        tok.bos_to_prepend().map_err(|e| engine_error(&e))?
    } else {
        None
    };
    ::embed::input::prompt_ids(
        k,
        v,
        |text| tok.encode(text),
        bos,
        (special && tok.add_eos()).then(|| tok.eos()),
        tok.n_vocab(),
    )
    .map_err(|e| ApiError::from(e).at(k))
}

/// Whether a prompt object carries media: a `multimodal_data` that holds
/// anything (`null` or an empty array, string or object holds nothing).
use ::embed::input::carries_media;

/// The last `/completion` object. `tokens` (the generated ids, the
/// end-of-generation one included) is llama.cpp's `return_tokens` field. A
/// whole answer that asked for probabilities carries every generated token's
/// entry in `completion_probabilities`; a stream's came with its chunks.
fn completion_final(
    state: &State,
    o: &Outcome,
    p: &GenParams,
    prompt: &Value,
    return_tokens: bool,
    slot: usize,
) -> Value {
    let mut v = json!({
        "content": if p.stream { "" } else { o.content.as_str() },
        "generated_text": o.content,
        "id_slot": slot,
        "stop": true,
        "model": state.alias,
        "tokens_predicted": o.timings.predicted_n,
        "tokens_evaluated": o.timings.n_prompt,
        "generation_settings": generation_settings(state, p),
        "prompt": prompt,
        "truncated": o.truncated,
        "stopped_eos": o.stop == genloop::StopKind::Eos,
        "stopped_word": o.stop == genloop::StopKind::Word,
        "stopped_limit": o.stop == genloop::StopKind::Limit,
        "stopping_word": o.stopping_word,
        "stop_type": o.stop.as_str(),
        "tokens_cached": o.timings.n_past,
        "timings": o.timings.to_json(),
    });
    if return_tokens {
        v["tokens"] = json!(o.tokens);
    }
    if !p.stream
        && let Some(ask) = &p.logprobs
    {
        v["completion_probabilities"] = Value::Array(ask.whole(&*state.tok));
    }
    v
}

fn sse(s: &mut EventStream<'_>, v: &Value) -> io::Result<()> {
    s.send(format!("data: {v}\n\n").as_bytes())
}

/// A 500 carrying `e`'s message.
fn engine_error(e: &dyn fmt::Display) -> ApiError {
    ApiError {
        retry_after: false,
        param: None,
        code: 500,
        kind: "server_error",
        message: e.to_string(),
    }
}

/// What a completion request runs, built by [`completion_plan`].
struct CompletionPlan {
    p: GenParams,
    /// The prompt as the engine is fed it.
    input: Prompt,
    /// The request's `prompt`, as the answer echoes it.
    prompt: Value,
    /// `return_tokens`: the answer carries the generated ids.
    return_tokens: bool,
}

/// The completion path's steps on a completion body: the sampling fields,
/// the prompt's ids ([`completion_prompt`]) and the think-span budget and
/// loop guard, whose span check reads the prompt's text (a string prompt is
/// its own, an id array's its decode) and whose span ends where the chat
/// parser's does (a Qwen3-Coder call ends it too). Every route that runs a
/// completion builds its generation here.
fn completion_plan(state: &State, b: &Map<String, Value>) -> Result<CompletionPlan, ApiError> {
    let mut p = gen_params(state, b)?;
    let ids = completion_prompt(state, b.get("prompt"))?;
    let call = follows_reasoning(&p);
    if p.reasoning_budget.is_some() || p.reasoning_loop || call {
        let text = match b.get("prompt") {
            Some(Value::String(s)) => s.clone(),
            _ => state.tok.decode(&ids),
        };
        gate_think_span(&mut p, &text);
        // A forced call's span ends at its close alone ([`ChatParser::following_reasoning`]).
        if let (ToolFormat::QwenXml(framing), false) = (state.tool_format, call) {
            p.think_split = ChatParser::with_tags(
                &text,
                ReasoningFormat::None,
                Some(Tools::QwenXml(framing, ParamKinds::default())),
                state.think,
            )
            .reading_specials(&*state.tok)
            .think;
        }
    }
    Ok(CompletionPlan {
        p,
        input: Prompt::from(ids),
        prompt: b.get("prompt").cloned().unwrap_or(Value::Null),
        return_tokens: get_b(b, "return_tokens")?.unwrap_or(false),
    })
}

/// [`completion_plan`] with the probabilities `/completion` renders:
/// `n_probs`, taken out of `b` first ([`logprobs::completion_ask`]) so the
/// shared steps' refusal stays every other route's; the plan's generation
/// carries what was asked.
fn completion_plan_logprobs(
    state: &State,
    b: &mut Map<String, Value>,
) -> Result<CompletionPlan, ApiError> {
    let ask = logprobs::completion_ask(b, state.info.n_vocab)?;
    let mut plan = completion_plan(state, b)?;
    plan.p.logprobs = ask;
    Ok(plan)
}

fn completion(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let plan = match gen_body(req).and_then(|mut b| completion_plan_logprobs(state, &mut b)) {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let CompletionPlan {
        p,
        input,
        prompt,
        return_tokens,
    } = plan;
    if !p.stream {
        return match run_whole(state, &input, prompt.clone(), &p, req, w)? {
            Err(e) => send_error(w, req, &e),
            Ok((o, slot)) => send_json(
                w,
                req,
                200,
                &completion_final(state, &o, &p, &prompt, return_tokens, slot),
            ),
        };
    }
    let peer = w.try_clone()?;
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    // The tokens whose entries went out with a chunk: each chunk carries the
    // tokens its event counts past them.
    let mut sent = 0;
    let r = {
        let mut sink = |ev: Event<'_>, slot: usize| -> io::Result<()> {
            if stream.is_none() {
                let w = w_opt
                    .take()
                    .ok_or_else(|| io::Error::other("stream writer taken"))?;
                stream = Some(EventStream::start(w, req, 200, "text/event-stream")?);
            }
            let s = stream
                .as_mut()
                .ok_or_else(|| io::Error::other("no stream"))?;
            let v = match ev {
                Event::Prompt(t) => json!({
                    "content": "", "stop": false, "id_slot": slot, "multimodal": false,
                    "prompt_progress": progress(t),
                }),
                Event::Text(text, _, t) => {
                    let mut v = json!({ "content": text, "stop": false, "id_slot": slot, "multimodal": false });
                    if tpt {
                        v["timings"] = t.to_json();
                    }
                    if let Some(ask) = &p.logprobs {
                        let entries = ask.entries(sent, t.predicted_n, &*state.tok);
                        sent = t.predicted_n;
                        if !entries.is_empty() {
                            v["completion_probabilities"] = Value::Array(entries);
                        }
                    }
                    v
                }
            };
            sse(s, &v)
        };
        run_gen(
            state,
            &req.path,
            &input,
            prompt.clone(),
            &p,
            Client::of(peer, req),
            &mut sink,
        )
    };
    finish_stream(req, stream, w_opt, r, |s, (o, slot)| {
        sse(
            s,
            &completion_final(state, o, &p, &prompt, return_tokens, *slot),
        )
    })
}

/// The one `prompt_progress` this server sends, once the prompt is evaluated:
/// `processed` is then what the cache holds of the prompt, all of it, as
/// llama-server's last progress event counts it (`slot.prompt.tokens.size()`).
/// `time_ms` is `prompt_ms`, which leaves the prompt cache's work
/// (`cache_ms`) out, as llama-server's progress clock starts after its slot's
/// cache work.
fn progress(t: &Timings) -> Value {
    json!({ "total": t.n_prompt, "cache": t.cache_n, "processed": t.n_prompt, "time_ms": t.prompt_ms })
}

/// Ends a stream: validation errors before the first byte go out as a plain
/// error response; an engine error mid-stream as an `error` event.
fn finish_stream<T>(
    req: &Request,
    stream: Option<EventStream<'_>>,
    w_opt: Option<&mut TcpStream>,
    r: Result<Result<T, GenError>, ApiError>,
    last: impl FnOnce(&mut EventStream<'_>, &T) -> io::Result<()>,
) -> io::Result<bool> {
    let mut stream = match (stream, w_opt) {
        (Some(s), _) => s,
        (None, Some(w)) => match &r {
            Err(e) => return send_error(w, req, e),
            Ok(Err(e)) => return send_error(w, req, &engine_error(e)),
            Ok(Ok(_)) => EventStream::start(w, req, 200, "text/event-stream")?,
        },
        (None, None) => return Ok(false),
    };
    match r {
        Ok(Ok(o)) => last(&mut stream, &o)?,
        Ok(Err(GenError::Client(e))) => return Err(e),
        Ok(Err(e)) => {
            let err = engine_error(&e);
            sse(&mut stream, &err.body())?;
        }
        Err(e) => sse(&mut stream, &e.body())?,
    }
    let reusable = stream.reusable();
    stream.finish()?;
    Ok(reusable)
}

struct ChatIds {
    id: String,
    created: u64,
    model: String,
}

impl ChatIds {
    /// `call_<index>_<request nonce>`: unique per call and per request, the same
    /// in every chunk that names the call.
    fn call_id(&self, index: usize) -> String {
        let nonce = self.id.strip_prefix("chatcmpl-").unwrap_or(&self.id);
        format!("call_{index}_{}", nonce.get(..16).unwrap_or(nonce))
    }

    fn tool_call(&self, c: &ToolCall) -> Value {
        json!({
            "id": self.call_id(c.index),
            "type": "function",
            "function": { "name": c.name, "arguments": c.arguments },
        })
    }

    /// The stream deltas for one parser step, in llama-server's order: reasoning,
    /// content, then the tool calls' pieces ([`CallPiece`]): a call's first
    /// delta carries its `index`, `id`, `type` and `function.name`, the next
    /// ones its `function.arguments` fragments.
    fn deltas(&self, d: &Message, pieces: &[CallPiece]) -> Vec<Value> {
        let mut out = Vec::new();
        if !d.reasoning.is_empty() {
            out.push(json!({ "reasoning_content": d.reasoning }));
        }
        if !d.content.is_empty() {
            out.push(json!({ "content": d.content }));
        }
        for piece in pieces {
            let call = match piece {
                CallPiece::Open { index, name } => json!({
                    "index": index,
                    "id": self.call_id(*index),
                    "type": "function",
                    "function": { "name": name, "arguments": "" },
                }),
                CallPiece::Args { index, text } => json!({
                    "index": index,
                    "function": { "arguments": text },
                }),
            };
            out.push(json!({ "tool_calls": [call] }));
        }
        out
    }

    fn chunk(&self, choices: Value) -> Value {
        json!({
            "choices": choices,
            "created": self.created,
            "id": self.id,
            "model": self.model,
            "object": "chat.completion.chunk",
        })
    }
}

/// OpenAI `finish_reason`: `tool_calls` when a call parsed and the generation
/// ended on EOS or a stop word (llama-server's rule).
fn chat_finish_reason(o: &Outcome, m: &Message) -> &'static str {
    match o.stop.finish_reason() {
        "stop" if !m.calls.is_empty() => "tool_calls",
        r => r,
    }
}

/// The scan a chat request's output gets: none, or the markup of the
/// server's template (Muse Spark's, which also splits its reasoning, on every
/// turn); a request that asks for tool calls under a template
/// whose markup no parser reads is refused, never answered with the calls
/// left in `content`.
fn tool_scan(state: &State, b: &Map<String, Value>) -> Result<Option<Tools>, ApiError> {
    if !parses_tools(b) {
        // Muse Spark's reasoning is a message to `self`, which only its recipient scan splits out
        // (the template names no think tag), so a turn with no tools is scanned too.
        return Ok(matches!(state.tool_format, ToolFormat::MuseSpark).then_some(Tools::MuseSpark));
    }
    Ok(Some(match state.tool_format {
        ToolFormat::Dsml(spelling) => Tools::Dsml(spelling),
        ToolFormat::KvMarkup(spelling) => {
            Tools::KvMarkup(spelling, ArgTypes::of_tools(b.get("tools")))
        }
        ToolFormat::Hermes => Tools::Hermes,
        ToolFormat::QwenXml(framing) => {
            Tools::QwenXml(framing, ParamKinds::of_tools(b.get("tools")))
        }
        ToolFormat::MuseSpark => Tools::MuseSpark,
        ToolFormat::Unparsed => {
            return Err(ApiError {
                retry_after: false,
                param: None,
                code: 501,
                kind: "not_supported_error",
                message: "the chat template's tool-call markup has no parser in this server \
                          (DSML, GLM's and Solar's key/value markup, Hermes' and Qwen's XML, \
                          and Muse's recipient markup are parsed): send the request without \
                          tools or with tool_choice \"none\""
                    .to_owned(),
            });
        }
    }))
}

/// A generation whose tool-call markup does not parse: the server's error,
/// as llama-server answers output its chat parser refuses.
fn tool_markup_error(e: &MarkupError) -> ApiError {
    ApiError {
        retry_after: false,
        param: None,
        code: 500,
        kind: "server_error",
        message: format!("tool-call markup: {e}"),
    }
}

/// OpenAI's `usage` of a request's choices (each outcome and its slot): the
/// prompt once, as the first choice evaluated it, and every choice's
/// completion tokens.
fn usage(all: &[(Outcome, usize)]) -> Value {
    let completion: usize = all.iter().map(|(o, _)| o.timings.predicted_n).sum();
    let prompt = all.first().map(|(o, _)| &o.timings);
    let n_prompt = prompt.map_or(0, |t| t.n_prompt);
    json!({
        "completion_tokens": completion,
        "prompt_tokens": n_prompt,
        "total_tokens": completion + n_prompt,
        "prompt_tokens_details": { "cached_tokens": prompt.map_or(0, |t| t.cache_n) },
    })
}

/// The choices an OpenAI request asks for, its `n` (1 when unset), taken out
/// of `b` first so the shared steps' refusal of `n > 1` stays every other
/// route's. A request's choices queue together ([`Board::admits`]), so they
/// are at most the slots and the requests that may wait beside them; a count
/// below 1 or past that is a 400 naming `n`.
fn take_choices(state: &State, b: &mut Map<String, Value>) -> Result<usize, ApiError> {
    let most = state.parallel + state.depth;
    let n = get_i(b, "n")?.unwrap_or(1);
    b.remove("n");
    usize::try_from(n)
        .ok()
        .filter(|n| (1..=most).contains(n))
        .ok_or_else(|| {
            invalid(format!(
                "n must be between 1 and {most}, the {} slots and the {} requests that may wait \
                 beside them, not {n}",
                state.parallel, state.depth
            ))
            .at("n")
        })
}

/// The generation of each of `n` choices of `p`: choice `i` samples with
/// `p`'s seed plus `i`, so choice 0 is the request of one choice and a
/// seeded request repeats choice by choice, and logs its own probabilities.
fn choice_params(state: &State, p: &GenParams, n: usize) -> Vec<GenParams> {
    (0..n as u64)
        .map(|i| {
            let mut c = p.clone();
            c.sampling.seed = p.sampling.seed.wrapping_add(i);
            c.logprobs = p
                .logprobs
                .as_ref()
                .map(|a| logprobs::Ask::new(a.asked(), state.info.n_vocab));
            c
        })
        .collect()
}

/// What a chat request runs, built by [`chat_plan`].
struct ChatPlan {
    p: GenParams,
    /// The prompt as the engine is fed it.
    input: Prompt,
    /// The rendered prompt's text, as the answer's `prompt` echoes it.
    prompt: Value,
    /// The parser the output goes through.
    parser: ChatParser,
}

/// The prompt a chat body renders, by the chat template, and what the engine
/// is fed: its ids, each image expanded to its span ([`chat_prompt`]). Every
/// route that counts a chat request's input tokens counts these ids.
fn chat_input(state: &State, b: &Map<String, Value>) -> Result<(String, Prompt), ApiError> {
    let rendered = render_chat(state, b)?;
    let prompt = chat_prompt(state, &rendered)?;
    Ok((rendered.text, prompt))
}

/// The chat path's steps on a chat body, in one order, so the first error is
/// the same whichever API sent the request: the sampling fields, the
/// reasoning format (`format`, the request's `reasoning_format` on the chat
/// path; `None` takes the default), the prompt ([`chat_input`]), the
/// tool-call scan and the think-span budget and loop guard. The chat path and every API
/// converted onto it build their generation here.
fn chat_plan(
    state: &State,
    b: &Map<String, Value>,
    format: Option<&Value>,
) -> Result<ChatPlan, ApiError> {
    let mut p = gen_params(state, b)?;
    let format =
        ReasoningFormat::from_request(format).map_err(|e| invalid(e).at("reasoning_format"))?;
    let (text, input) = chat_input(state, b)?;
    let tools = tool_scan(state, b)?;
    gate_think_span(&mut p, &text);
    let mut parser =
        ChatParser::with_tags(&text, format, tools, state.think).for_template(state.tool_format);
    if follows_reasoning(&p) {
        parser = parser.following_reasoning();
    }
    let parser = parser.reading_specials(&*state.tok);
    // Without a reasoning split or tool scan, the guard still tracks the server's think tags.
    p.think_split.clone_from(&parser.think);
    Ok(ChatPlan {
        p,
        input,
        prompt: Value::String(text),
        parser,
    })
}

/// [`chat_plan`] with the probabilities the chat path renders: `logprobs` and
/// `top_logprobs`, taken out of `b` first ([`logprobs::chat_ask`]) so the
/// shared steps' refusal stays every other route's; the request's own
/// `reasoning_format`; the plan's generation carries what was asked.
fn chat_plan_logprobs(state: &State, b: &mut Map<String, Value>) -> Result<ChatPlan, ApiError> {
    let ask = logprobs::chat_ask(b, state.info.n_vocab)?;
    let b: &Map<String, Value> = b;
    let mut plan = chat_plan(state, b, b.get("reasoning_format"))?;
    plan.p.logprobs = ask;
    Ok(plan)
}

fn chat(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = gen_body(req).and_then(|mut b| {
        let n = take_choices(state, &mut b)?;
        let plan = chat_plan_logprobs(state, &mut b)?;
        Ok((b, n, plan))
    });
    let (b, n, plan) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let ChatPlan {
        p,
        input,
        prompt,
        parser,
    } = plan;
    let parser = if p.stream {
        parser.streaming_calls()
    } else {
        parser.keeping_cut_call()
    };
    // Each choice parses its own output.
    let mut parsers = vec![parser; n];
    let ps = choice_params(state, &p, n);
    let ids_meta = ChatIds {
        id: format!("chatcmpl-{}", state.random_id()),
        created: unix_now(),
        model: b
            .get("model")
            .and_then(Value::as_str)
            .map_or_else(|| state.alias.clone(), str::to_owned),
    };
    let vocab = &*state.tok;
    if !p.stream {
        let all = match run_whole_choices(state, &input, prompt, &ps, req, w)? {
            Ok(all) => all,
            Err(e) => return send_error(w, req, &e),
        };
        let mut choices = Vec::with_capacity(n);
        for (i, ((o, _), parser)) in all.iter().zip(&mut parsers).enumerate() {
            if let Err(e) = parser
                .try_push(&o.content, &o.specials)
                .and_then(|_| parser.try_finish_at(o.cut()))
            {
                return send_error(w, req, &tool_markup_error(&e));
            }
            let lp = ps[i]
                .logprobs
                .as_ref()
                .map(|a| logprobs::chat_object(a.whole(vocab)));
            choices.push(chat_choice(&ids_meta, i, o, parser.message(), lp));
        }
        return send_json(w, req, 200, &chat_final(&ids_meta, choices, &all));
    }
    let peer = w.try_clone()?;
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    // Markup that does not parse stops the generation (the sink fails) and
    // ends the stream with an error event instead of a dropped connection.
    let mut markup: Option<MarkupError> = None;
    // The tokens whose entries went out, a choice's own: a chunk's last delta
    // carries the tokens its event counts past them; an event that makes no
    // delta leaves its tokens to the next chunk that has one, the last chunk
    // what is left.
    let mut sent = vec![0; n];
    let r = {
        let meta = &ids_meta;
        let parsers = &mut parsers;
        let ps = &ps;
        let mut sink = |ev: Event<'_>, i: usize, _slot: usize| -> io::Result<()> {
            if stream.is_none() {
                let w = w_opt
                    .take()
                    .ok_or_else(|| io::Error::other("stream writer taken"))?;
                let mut s = EventStream::start(w, req, 200, "text/event-stream")?;
                open_choices(&mut s, meta, n)?;
                stream = Some(s);
            }
            let s = stream
                .as_mut()
                .ok_or_else(|| io::Error::other("no stream"))?;
            match ev {
                Event::Prompt(t) => {
                    let mut v = meta.chunk(json!([]));
                    v["prompt_progress"] = progress(t);
                    sse(s, &v)
                }
                Event::Text(text, specials, t) => {
                    let parser = &mut parsers[i];
                    let d = match parser.try_push(text, specials) {
                        Ok(d) => d,
                        Err(e) => {
                            markup = Some(e);
                            return Err(io::Error::other("tool-call markup does not parse"));
                        }
                    };
                    let deltas = meta.deltas(&d, &parser.take_pieces());
                    let last = deltas.len();
                    for (k, delta) in deltas.into_iter().enumerate() {
                        let mut v = meta.chunk(json!([{
                            "finish_reason": null, "index": i, "delta": delta,
                        }]));
                        if k + 1 == last
                            && let Some(ask) = &ps[i].logprobs
                        {
                            let entries = ask.entries(sent[i], t.predicted_n, vocab);
                            sent[i] = t.predicted_n;
                            if !entries.is_empty() {
                                v["choices"][0]["logprobs"] = logprobs::chat_object(entries);
                            }
                        }
                        if tpt {
                            v["timings"] = t.to_json();
                        }
                        sse(s, &v)?;
                    }
                    Ok(())
                }
            }
        };
        run_choices(
            state,
            &req.path,
            &input,
            prompt,
            ps,
            Client::of(peer, req),
            &mut sink,
        )
    };
    let r = match markup {
        Some(e) => Err(tool_markup_error(&e)),
        None => r,
    };
    let include_usage = p.include_usage;
    let opened = stream.is_some();
    finish_stream(req, stream, w_opt, r, |s, all| {
        if !opened {
            open_choices(s, &ids_meta, n)?;
        }
        let mut last = Value::Null;
        for (i, ((o, _), parser)) in all.iter().zip(&mut parsers).enumerate() {
            // The choice before this one finishes first.
            if !last.is_null() {
                sse(s, &last)?;
            }
            let d = match parser.try_finish_at(o.cut()) {
                Ok(d) => d,
                Err(e) => {
                    let e = tool_markup_error(&e);
                    return sse(s, &e.body());
                }
            };
            // The tokens no chunk carried yet go with the last delta, or with
            // the finishing chunk when the parser releases none.
            let mut left = ps[i]
                .logprobs
                .as_ref()
                .map(|a| a.entries(sent[i], o.timings.predicted_n, vocab))
                .filter(|e| !e.is_empty());
            let deltas = ids_meta.deltas(&d, &parser.take_pieces());
            let count = deltas.len();
            for (k, delta) in deltas.into_iter().enumerate() {
                let mut v =
                    ids_meta.chunk(json!([{ "finish_reason": null, "index": i, "delta": delta }]));
                if k + 1 == count
                    && let Some(e) = left.take()
                {
                    v["choices"][0]["logprobs"] = logprobs::chat_object(e);
                }
                sse(s, &v)?;
            }
            last = ids_meta.chunk(json!([{
                "finish_reason": chat_finish_reason(o, parser.message()), "index": i, "delta": {},
            }]));
            if let Some(e) = left.take() {
                last["choices"][0]["logprobs"] = logprobs::chat_object(e);
            }
        }
        if include_usage {
            sse(s, &last)?;
            last = ids_meta.chunk(json!([]));
            last["usage"] = usage(all);
        }
        if let Some((o, _)) = all.first() {
            last["timings"] = o.timings.to_json();
        }
        sse(s, &last)?;
        s.send(b"data: [DONE]\n\n")
    })
}

/// The chunks that open a chat stream: OpenAI opens each choice with its role.
fn open_choices(s: &mut EventStream<'_>, meta: &ChatIds, n: usize) -> io::Result<()> {
    for i in 0..n {
        sse(
            s,
            &meta.chunk(json!([{
                "finish_reason": null, "index": i,
                "delta": { "role": "assistant", "content": null },
            }])),
        )?;
    }
    Ok(())
}

/// One choice of the non-stream response, `index` its place among the
/// request's. `reasoning_content` and `tool_calls` appear only when
/// non-empty, as llama-server writes them; `logprobs`, the chat path's
/// probabilities object, only when the request asked for it.
fn chat_choice(
    meta: &ChatIds,
    index: usize,
    o: &Outcome,
    m: &Message,
    probs: Option<Value>,
) -> Value {
    let mut message = json!({ "role": "assistant", "content": m.content });
    if !m.reasoning.is_empty() {
        message["reasoning_content"] = json!(m.reasoning);
    }
    if !m.calls.is_empty() {
        message["tool_calls"] = m.calls.iter().map(|c| meta.tool_call(c)).collect();
    }
    let mut choice = json!({
        "finish_reason": chat_finish_reason(o, m),
        "index": index,
        "message": message,
    });
    if let Some(lp) = probs {
        choice["logprobs"] = lp;
    }
    choice
}

/// The non-stream response: its `choices`, the `usage` of all of them
/// ([`usage`]) and the first choice's `timings`.
fn chat_final(meta: &ChatIds, choices: Vec<Value>, all: &[(Outcome, usize)]) -> Value {
    json!({
        "choices": choices,
        "created": meta.created,
        "model": meta.model,
        "object": "chat.completion",
        "usage": usage(all),
        "id": meta.id,
        "timings": all.first().map(|(o, _)| o.timings.to_json()),
    })
}

/// The fixture chat templates the tests below read, shared with `tests/qwenxml.rs`.
#[cfg(test)]
#[path = "../tests/common/fixture_templates.rs"]
mod fixture_templates;

/// The chat path's control-string gates: client text tokenizes as text on every fixture family.
#[cfg(test)]
#[path = "ctl_tests.rs"]
mod ctl_tests;

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        ACCEPT_BACKOFF, EBADF, EFAULT, EINVAL, EMFILE, ENFILE, End, Engine, EngineProps, Park,
        ServeError, Server, SlotConfig, SlotQueue, Stop, ThinkingOff, after_accept_error,
        argv_redacted, carries_media, check_slots, content_text, deny_line, engine_object, id_half,
        media_error, needs_object_arguments, probe_thinking_off, shutdown_allowed,
    };
    use crate::engine::{DeviceProps, PlacementProps, SamplerRefused, SamplingParams};
    use crate::flag::{ApiKeys, Denied};
    use crate::media::{ExpandError, MediaError};
    use crate::reasoning::ThinkTags;
    use jinja::ChatTemplate;
    use serde_json::{Value, json};
    use std::io::{self, Read};
    use std::net::SocketAddr;
    use std::time::{Duration, Instant};

    /// The 400 for a prompt past the context names the `--ctx-size` that
    /// holds the prompt and the generation asked [FAIL-first: the message
    /// ends at the context's number]: the smallest power of two at least the
    /// prompt plus the request's positive `n_predict` (one token when it sets
    /// none or 0), and under several slots the flag a seat that splits it
    /// needs.
    #[test]
    fn the_over_context_hint_names_the_flag_that_holds_the_request() {
        use super::ctx_hint;
        let rows: [(usize, i64, usize, &str); 8] = [
            (5600, -1, 1, "; serve it with --ctx-size 8192"),
            (5600, 4000, 1, "; serve it with --ctx-size 16384"),
            (4096, -1, 1, "; serve it with --ctx-size 8192"),
            (4096, 0, 1, "; serve it with --ctx-size 8192"),
            (4095, -1, 1, "; serve it with --ctx-size 4096"),
            (4095, 1, 1, "; serve it with --ctx-size 4096"),
            (4095, 2, 1, "; serve it with --ctx-size 8192"),
            (
                5600,
                -1,
                2,
                "; serve it with --ctx-size 8192 (under --parallel 2, where the slots split the \
                 flag: --ctx-size 16384)",
            ),
        ];
        for (prompt, n_predict, slots, want) in rows {
            assert_eq!(
                ctx_hint(prompt, n_predict, slots),
                want,
                "{prompt} tokens, n_predict {n_predict}, {slots} slots"
            );
        }
        assert_eq!(
            ctx_hint(usize::MAX, i64::MAX, 1),
            "",
            "no power of two fits"
        );
    }

    /// Text parts join by newline, as one string `content` would carry them.
    /// Every other part is refused by name, wherever it stands in the array:
    /// a media part by its kind in llama-server's words, a part of any other
    /// type (or none) as an unsupported type, a text part whose `text` is not
    /// a string as such. Nothing is dropped.
    #[test]
    fn content_parts_join_text_and_refuse_the_rest_by_name() {
        let text = |t: &str| json!({"type": "text", "text": t});
        assert_eq!(
            content_text(&[text("a"), text("b")]).ok().as_deref(),
            Some("a\nb")
        );
        assert_eq!(content_text(&[text("hi")]).ok().as_deref(), Some("hi"));
        assert_eq!(content_text(&[]).ok().as_deref(), Some(""));
        let image = "image input is not supported: this server reads text only";
        let audio = "audio input is not supported: this server reads text only";
        let video = "video input is not supported: this server reads text only";
        let unsupported = "unsupported content[].type";
        let not_string = "content[].text must be a string";
        let refused = [
            (
                json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}),
                image,
            ),
            (
                json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}),
                audio,
            ),
            (
                json!({"type": "input_video", "input_video": {"url": "file:///v.mp4"}}),
                video,
            ),
            (
                json!({"type": "video_url", "video_url": {"url": "file:///v.mp4"}}),
                video,
            ),
            (json!({"type": "bogus"}), unsupported),
            (json!({"type": ""}), unsupported),
            (json!({"type": 7}), unsupported),
            (json!({"text": "a"}), unsupported),
            (json!("a"), unsupported),
            (json!({"type": "text", "text": 3}), not_string),
            (json!({"type": "text", "text": null}), not_string),
            (json!({"type": "text"}), not_string),
        ];
        for (part, message) in refused {
            for parts in [vec![part.clone()], vec![text("a"), part.clone(), text("b")]] {
                match content_text(&parts) {
                    Ok(t) => panic!("{part} accepted as {t:?}"),
                    Err(e) => {
                        assert_eq!((e.code, e.kind), (400, "invalid_request_error"), "{part}");
                        assert_eq!(e.message, message, "{part}");
                    }
                }
            }
        }
    }

    /// A `/completion` prompt object carries media when its `multimodal_data`
    /// holds anything; `null`, empty or absent carries none, and a prompt that
    /// is not an object carries none.
    #[test]
    fn prompt_media_is_a_non_empty_multimodal_data() {
        let with = |d: Value| json!({"prompt_string": "ab", "multimodal_data": d});
        for d in [json!(["AAAA"]), json!("AAAA"), json!({"a": 1}), json!(0)] {
            assert!(carries_media(&with(d.clone())), "{d}");
        }
        for d in [json!([]), json!(""), json!({}), Value::Null] {
            assert!(!carries_media(&with(d.clone())), "{d}");
        }
        for v in [
            json!({"prompt_string": "ab"}),
            json!("ab"),
            json!([1, 2]),
            json!(7),
        ] {
            assert!(!carries_media(&v), "{v}");
        }
    }

    /// Ollama's `/api/show` of the chat model: its template, what the file
    /// states (format, arch, quant), the served context under the arch, and
    /// `tools` and `vision` for a template whose tool markup the server parses
    /// and an engine that takes images.
    #[test]
    fn show_names_the_files_facts_and_capabilities() {
        struct Filed(crate::MockEngine);
        impl Engine for Filed {
            fn tokenizer(&self) -> Arc<dyn crate::Tokenizer> {
                self.0.tokenizer()
            }
            fn media_model(&self) -> Option<crate::media::SharedMediaModel> {
                self.0.media_model()
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<(), crate::EngineError> {
                self.0.prefill(ids)
            }
            fn next(
                &mut self,
                last: u32,
                out: Option<&mut [f32]>,
            ) -> Result<u32, crate::EngineError> {
                self.0.next(last, out)
            }
            fn reset(&mut self) -> Result<(), crate::EngineError> {
                self.0.reset()
            }
            fn ctx_max(&self) -> usize {
                self.0.ctx_max()
            }
            fn describe(&self) -> String {
                self.0.describe()
            }
            fn props_engine(&self) -> EngineProps {
                let model = crate::ModelProps {
                    format: Some("gguf".to_owned()),
                    arch: Some("qwen38".to_owned()),
                    quant: Some("q4_0".to_owned()),
                    ..crate::ModelProps::default()
                };
                EngineProps {
                    model: Some(model),
                    ..EngineProps::default()
                }
            }
        }
        let mut config = super::testserve::mock_config();
        config.chat_template = format!("{}{{#- to=commentary <|eom|> #}}", config.chat_template);
        let template = config.chat_template.clone();
        let engine = Filed(crate::MockEngine::new(64).with_media().0);
        let (addr, _state, _end) = super::testserve::spawn(Box::new(engine), config);
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/api/show", &[], r#"{"model":"mock"}"#);
        assert_eq!(status, 200, "{body}");
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap(),
            json!({
                "template": template,
                "details": {
                    "format": "gguf",
                    "family": "qwen38",
                    "families": ["qwen38"],
                    "quantization_level": "q4_0",
                },
                "model_info": { (gguf::GENERAL_ARCHITECTURE): "qwen38", "qwen38.context_length": 64 },
                "capabilities": ["completion", "tools", "vision"],
            })
        );
    }

    /// `engine.ctx_verified` is the engine's verified context when it
    /// reports one, and absent when it does not: nothing is guessed.
    #[test]
    fn engine_object_carries_the_verified_context_only_when_reported() {
        let with = engine_object(&EngineProps {
            ctx_verified: Some(3001),
            ..EngineProps::default()
        });
        assert_eq!(with["ctx_verified"], 3001, "{with}");
        let without = engine_object(&EngineProps::default());
        assert!(without.get("ctx_verified").is_none(), "{without}");
    }

    /// Each device's `bytes` is the sum of its classes, a device with none
    /// sums to 0, and the classes and the cards' KV bytes stand beside it:
    /// the one place the `/props` placement renders its totals.
    #[test]
    fn placement_bytes_are_the_class_sums() {
        let device = |name: &str, classes: &[(&str, u64)]| DeviceProps {
            device: name.to_owned(),
            class_bytes: classes.iter().map(|&(k, b)| (k.to_owned(), b)).collect(),
            ..DeviceProps::default()
        };
        let engine = engine_object(&EngineProps {
            placement: Some(PlacementProps {
                devices: vec![
                    device("GPU0", &[("dense", 7), ("experts", 30), ("kv", 5)]),
                    device("CPU", &[("experts", 100)]),
                    device("GPU1", &[]),
                ],
                vram_kv_bytes: Some(5),
            }),
            ..EngineProps::default()
        });
        let devices = engine["placement"]["devices"].as_array().expect("devices");
        let bytes: Vec<(&str, u64)> = devices
            .iter()
            .map(|d| {
                (
                    d["device"].as_str().expect("a name"),
                    d["bytes"].as_u64().expect("bytes"),
                )
            })
            .collect();
        assert_eq!(bytes, [("GPU0", 42), ("CPU", 100), ("GPU1", 0)], "{engine}");
        assert_eq!(
            devices[0]["classes"],
            json!({"dense": 7, "experts": 30, "kv": 5})
        );
        assert_eq!(engine["placement"]["vram_kv_bytes"], 5);
        assert!(
            engine_object(&EngineProps::default())
                .get("placement")
                .is_none()
        );
    }

    /// A failed `accept` ends the server only when the listener no longer
    /// stands or the error is the listening socket's own; a client's aborted
    /// connection retries at once, running out of descriptors or memory, or
    /// failing twice in a row, backs off.
    #[test]
    fn accept_failures_end_the_server_only_with_the_listener() {
        let aborted = io::Error::from(io::ErrorKind::ConnectionAborted);
        assert_eq!(after_accept_error(&aborted, true, 0), Some(Duration::ZERO));
        assert_eq!(after_accept_error(&aborted, true, 1), Some(ACCEPT_BACKOFF));
        for raw in [EMFILE, ENFILE] {
            let e = io::Error::from_raw_os_error(raw);
            assert_eq!(after_accept_error(&e, true, 0), Some(ACCEPT_BACKOFF), "{e}");
        }
        let oom = io::Error::from(io::ErrorKind::OutOfMemory);
        assert_eq!(after_accept_error(&oom, true, 0), Some(ACCEPT_BACKOFF));
        assert_eq!(after_accept_error(&aborted, false, 0), None);
        for raw in [EBADF, EFAULT, EINVAL] {
            let e = io::Error::from_raw_os_error(raw);
            assert_eq!(after_accept_error(&e, true, 0), None, "{e}");
        }
    }

    /// A try-draw never jumps ahead of queued tickets: while any handed-out
    /// ticket is unserved the queue answers `None`; idle, it answers the
    /// ticket a waiting caller would have drawn.
    #[test]
    fn try_draw_refuses_while_a_ticket_waits() {
        let q = SlotQueue::default();
        assert_eq!(q.try_draw(), Some(0), "idle: the ticket is the holder's");
        assert_eq!(q.draw(), 1, "drawn after the try-drawn ticket");
        assert_eq!(q.draw(), 2);
        assert_eq!(
            q.try_draw(),
            None,
            "two tickets wait: a try must not jump ahead"
        );
        q.pass(1);
        assert_eq!(q.try_draw(), None, "one ticket still waits");
        q.pass(2);
        assert_eq!(q.try_draw(), Some(3), "idle again after every turn passed");
    }

    /// The id's second half is the request counter under the process's seed:
    /// two seeds never draw the same half at the same request number — with
    /// the counter alone every process did.
    #[test]
    fn id_halves_differ_per_seed_at_the_same_counter() {
        let mut seen = std::collections::HashSet::new();
        for seed in 0..1000u64 {
            assert!(
                seen.insert(id_half(seed, 7)),
                "seed {seed} draws a half another seed drew at the same counter"
            );
        }
    }

    /// `POST /shutdown` is accepted only from a loopback peer: IPv4 loopback,
    /// `::1` and an IPv4-mapped loopback address pass; a LAN address, a
    /// public one and a mapped LAN one do not.
    #[test]
    fn shutdown_is_looped_back_only() {
        for ok in ["127.0.0.1", "127.255.255.254", "::1", "::ffff:127.0.0.1"] {
            let ip: std::net::IpAddr = ok.parse().unwrap();
            assert!(shutdown_allowed(ip), "{ip} is a loopback peer");
        }
        for no in [
            "192.168.1.5",
            "10.0.0.2",
            "172.16.0.9",
            "8.8.8.8",
            "2001:4860:4860::8888",
            "::ffff:192.168.1.5",
        ] {
            let ip: std::net::IpAddr = no.parse().unwrap();
            assert!(!shutdown_allowed(ip), "{ip} is not a loopback peer");
        }
    }

    /// One HTTP/1.0 request over a fresh connection, the integration
    /// harness's client shape copied in (a unit test cannot reach
    /// `tests/common`): the server closes after its answer, so the body is
    /// whatever arrives before that.
    fn roundtrip(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
        super::testserve::roundtrip(addr, method, path, &[], body)
    }

    /// The live view's copy of a server (`live_now`): a request that ended
    /// is listed with how it ended and its counts, its slot is idle again,
    /// the throughput is the `/metrics` totals', and a stop shows as one.
    #[test]
    fn the_live_view_copies_the_server() {
        use crate::live::{Ending, Phase, Status};
        let (addr, state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let now = super::live_now(&state);
        assert_eq!(now.status, Status::Ready);
        assert!(now.requests.is_empty());
        assert_eq!(now.room, state.room());

        let (status, body) = roundtrip(
            addr,
            "POST",
            super::path::COMPLETION,
            r#"{"prompt":"ab","n_predict":3}"#,
        );
        assert_eq!(status, 200, "{body}");
        let now = super::live_now(&state);
        assert_eq!(now.slots.len(), 1);
        assert_eq!(now.slots[0].phase, Phase::Idle);
        assert_eq!(now.requests.len(), 1, "the request that ended");
        let done = &now.requests[0];
        assert!(matches!(done.ending, Ending::Done(_)), "{:?}", done.ending);
        assert!(done.timings.predicted_n > 0 && done.timings.n_prompt > 0);
        let stats = super::relock(&state.shared.stats);
        assert!(now.decode_rate > 0.0);
        assert_eq!(now.decode_rate.to_bits(), stats.decode_rate().to_bits());
        assert_eq!(now.prompt_rate.to_bits(), stats.prompt_rate().to_bits());
        assert_eq!(
            (now.cached, now.evaluated),
            (stats.n_prompt_cached_total, stats.n_prompt_total)
        );
        drop(stats);

        let (status, body) = roundtrip(addr, "POST", "/shutdown", "");
        assert_eq!(status, 200, "{body}");
        assert_eq!(super::live_now(&state).status, Status::Stopping);
    }

    /// `POST /shutdown` over a real socket: the 200 with its body reaches the
    /// client first, the end channel carries the shutdown naming a loopback
    /// peer, a second post answers 200 and changes nothing, and a generation
    /// request after it is a 503 naming the shutdown (pinned: the listener
    /// stands while the stop runs — the exit belongs to `run`, which no test
    /// enters — so the refusal the client reads is the gate's 503, not a
    /// refused connection). The engine thread leaves its loop on its own; the
    /// engine it leaves is the process exit's.
    #[test]
    fn posted_shutdown_stops_the_server() {
        let (addr, state, ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );

        let (status, body) = roundtrip(addr, "POST", "/shutdown", "");
        assert_eq!(status, 200, "{body}");
        assert_eq!(body, r#"{"status":"shutting down"}"#, "the 200's body");

        match ended.recv_timeout(Duration::from_secs(5)) {
            Ok(End::Shutdown(Stop::Posted(peer))) => {
                assert!(peer.ip().is_loopback(), "the stop names its peer: {peer}")
            }
            Ok(_) => panic!("the end channel carried another end"),
            Err(e) => panic!("the end channel carries no shutdown: {e}"),
        }

        let (status, body) = roundtrip(addr, "POST", "/shutdown", "");
        assert_eq!(status, 200, "{body}");
        assert_eq!(body, r#"{"status":"shutting down"}"#);
        assert!(
            ended.recv_timeout(Duration::from_millis(200)).is_err(),
            "a second post changes nothing: no second end"
        );

        let chat = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":4}"#;
        let (status, body) = roundtrip(addr, "POST", "/v1/chat/completions", chat);
        assert_eq!(status, 503, "{body}");
        let v: Value = serde_json::from_str(&body).expect("the 503's body is JSON");
        let message = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("shutting down") && message.contains("no new request is admitted"),
            "the 503 names the shutdown: {message}"
        );

        assert!(
            state.shared.wait_loop_end(Duration::from_secs(5)),
            "the engine thread leaves its loop once the stop reaches it"
        );
    }

    /// `logprobs` asks for probabilities as a bool on the chat path and as a
    /// count on OpenAI's text completion path. `/completion` reads its
    /// probabilities from `n_probs`, so any `logprobs` but `false` is a 400
    /// naming the field there; the chat path serves the bool
    /// ([`super::logprobs`]'s tests) and refuses the count form by name;
    /// `false` and `null` are absent.
    #[test]
    fn logprobs_in_either_form_is_refused_by_name() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let chat = |lp: &str| {
            format!(
                r#"{{"messages":[{{"role":"user","content":"hi"}}],"max_tokens":1,"logprobs":{lp}}}"#
            )
        };
        let completion = |lp: &str| format!(r#"{{"prompt":"ab","max_tokens":1,"logprobs":{lp}}}"#);
        for (lp, path, body) in [
            ("true", "/completion", completion("true")),
            ("5", "/completion", completion("5")),
            ("0", "/completion", completion("0")),
            ("5", "/v1/chat/completions", chat("5")),
            ("0", "/v1/chat/completions", chat("0")),
        ] {
            let (status, text) = roundtrip(addr, "POST", path, &body);
            assert_eq!(status, 400, "{path} logprobs {lp}: {text}");
            assert!(
                text.contains("logprobs"),
                "{path} logprobs {lp}: the 400 names the field: {text}"
            );
        }
        for (path, body) in [
            ("/completion", completion("false")),
            ("/completion", completion("null")),
            ("/v1/chat/completions", chat("false")),
        ] {
            let (status, text) = roundtrip(addr, "POST", path, &body);
            assert_eq!(status, 200, "{path}: {text}");
        }
    }

    /// A vocabulary that names no BOS serves where it adds none: `/tokenize` puts nothing before a
    /// text and `bos_token` is empty text. Where it adds a BOS and names none, a text prompt is
    /// refused by name, not started with another id.
    #[test]
    fn a_vocabulary_with_no_bos_serves_only_where_it_adds_none() {
        for add_bos in [false, true] {
            let (addr, _state, _ended) = super::testserve::spawn(
                Box::new(crate::MockEngine::new(64).with_no_bos(add_bos)),
                super::testserve::mock_config(),
            );
            let (status, props) = roundtrip(addr, "GET", "/props", "");
            assert_eq!(status, 200, "{props}");
            let props: Value = serde_json::from_str(&props).expect("JSON");
            assert_eq!(props["bos_token"], "", "add_bos {add_bos}");
            let ids = |body: &str| roundtrip(addr, "POST", "/tokenize", body);
            let (status, plain) = ids(r#"{"content": "ab"}"#);
            assert_eq!(status, 200, "{plain}");
            let (status, special) = ids(r#"{"content": "ab", "add_special": true}"#);
            if add_bos {
                assert_eq!(status, 500, "{special}");
                assert!(special.contains("adds a BOS"), "{special}");
                assert!(special.contains("names no BOS token"), "{special}");
            } else {
                assert_eq!(status, 200, "{special}");
                assert_eq!(special, plain);
            }
        }
    }

    /// `/tokenize` takes `content` as a string or an array of strings and
    /// token ids, read as a `/completion` prompt is, with the BOS only under
    /// `add_special` and only before text; an absent `content` and any other
    /// type (`null` included) are a 400 naming it. `/detokenize` without
    /// `tokens` is a 400 naming it.
    #[test]
    fn tokenize_content_is_a_prompt_and_absent_is_a_400() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64).with_bos()),
            super::testserve::mock_config(),
        );
        let tokens = |body: Value| {
            let (status, text) = roundtrip(addr, "POST", "/tokenize", &body.to_string());
            assert_eq!(status, 200, "{body}: {text}");
            serde_json::from_str::<Value>(&text).expect("JSON")["tokens"].clone()
        };
        let bos = crate::engine::Tokenizer::bos(&crate::mock::BosTokenizer).expect("a BOS");
        let ab = tokens(json!({"content": "ab"}));
        let ab_ids: Vec<u64> = ab
            .as_array()
            .expect("tokens")
            .iter()
            .map(|x| x.as_u64().expect("an id"))
            .collect();
        assert!(!ab_ids.is_empty(), "{ab}");
        assert_eq!(tokens(json!({"content": ["ab"]})), ab);
        assert_eq!(tokens(json!({"content": ab_ids})), ab);
        assert_eq!(tokens(json!({"content": ["a", ab_ids[1]]})), ab);
        let mut with_bos = vec![bos as u64];
        with_bos.extend(&ab_ids);
        assert_eq!(
            tokens(json!({"content": "ab", "add_special": true})),
            json!(with_bos)
        );
        assert_eq!(
            tokens(json!({"content": ["ab"], "add_special": true})),
            json!(with_bos)
        );
        assert_eq!(
            tokens(json!({"content": ab_ids.clone(), "add_special": true})),
            ab,
            "no BOS before a prompt that starts with an id"
        );
        assert_eq!(
            tokens(json!({"content": "", "add_special": true})),
            json!([bos])
        );
        let refused = [
            ("/tokenize", json!({}), "'content' is required".to_string()),
            (
                "/tokenize",
                json!({"add_special": true}),
                "'content' is required".to_string(),
            ),
            ("/detokenize", json!({}), "'tokens' is required".to_string()),
            (
                "/tokenize",
                json!({"content": ["ab", {"multimodal_data": ["x"]}]}),
                "Multimodal data provided, but model does not support multimodal requests."
                    .to_string(),
            ),
        ]
        .into_iter()
        .chain(
            [
                (json!(5), "a number"),
                (json!({"text": "ab"}), "an object"),
                (json!(true), "a boolean"),
                (Value::Null, "null"),
            ]
            .into_iter()
            .map(|(content, got)| {
                (
                    "/tokenize",
                    json!({ "content": content }),
                    format!(
                        "content must be a string or an array of strings and token ids, not {got}"
                    ),
                )
            }),
        );
        for (path, body, want) in refused {
            let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
            assert_eq!(status, 400, "{path} {body}: {text}");
            assert_eq!(message_of(&text), want, "{path} {body}");
        }
    }

    #[test]
    fn generation_settings_keep_widened_f32_json_text() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let body = json!({
            "prompt": "ab", "n_predict": 1,
            "temperature": 0.8, "top_p": 0.9, "min_p": 0.1,
            "repeat_penalty": 1.1, "presence_penalty": -0.3, "frequency_penalty": 0.7,
            "logit_bias": {"7": 0.8, "8": -0.3},
        });
        let (status, text) = roundtrip(addr, "POST", "/completion", &body.to_string());
        assert_eq!(status, 200, "{text}");
        let reply: Value = serde_json::from_str(&text).expect("the completion is JSON");
        for (field, want) in [
            ("temperature", "0.800000011920929"),
            ("top_p", "0.8999999761581421"),
            ("min_p", "0.10000000149011612"),
            ("repeat_penalty", "1.100000023841858"),
            ("presence_penalty", "-0.30000001192092896"),
            ("frequency_penalty", "0.699999988079071"),
        ] {
            assert_eq!(
                reply["generation_settings"][field].to_string(),
                want,
                "{field}"
            );
        }
        assert_eq!(
            reply["generation_settings"]["logit_bias"].to_string(),
            r#"[{"bias":0.800000011920929,"token":7},{"bias":-0.30000001192092896,"token":8}]"#,
        );
    }

    /// A `temperature`, `top_p` or `min_p` whose `f32` is not finite (`1e39`
    /// is a finite JSON number and an infinite `f32`) is a 400 naming the
    /// field and "a finite number", where the sampler would refuse it and the
    /// request would run greedy. A finite value, whatever its range, is taken
    /// as it was, and a field of another type is a 400 naming it
    /// ([`a_field_of_another_type_is_a_400_naming_the_type_it_takes`]).
    #[test]
    fn non_finite_sampling_fields_are_refused_by_name() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let ask = |field: &str, value: &str| {
            let body = format!(r#"{{"prompt":"ab","n_predict":1,"{field}":{value}}}"#);
            roundtrip(addr, "POST", "/completion", &body)
        };
        for field in ["temperature", "top_p", "min_p"] {
            for value in ["1e39", "-1e39", "1e999"] {
                let (status, text) = ask(field, value);
                assert_eq!(status, 400, "{field} {value}: {text}");
                let v: Value = serde_json::from_str(&text).expect("the 400's body is JSON");
                let message = v["error"]["message"].as_str().unwrap_or_default();
                assert!(
                    message.contains(field) && message.contains("a finite number"),
                    "{field} {value}: the 400 names the field and the rule: {message}"
                );
            }
            // PIN(2026-10-09): a field of another type (`"hot"`) is a 400, no longer read as absent.
            for value in ["0.5", "-1", "3e38", "null"] {
                let (status, text) = ask(field, value);
                assert_eq!(status, 200, "{field} {value}: {text}");
            }
        }
    }

    /// The llama.cpp samplers this server lacks are a 400 naming them where
    /// llama-server would change a token, and pass where it would not (each
    /// field's off range in `llama-sampler.cpp`, including `top_n_sigma` 0);
    /// their secondary fields pass at any value of their type, except the
    /// values llama-server's schema refuses.
    #[test]
    fn samplers_this_server_lacks_are_refused_only_when_on() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let ask = |extra: Value| {
            let mut body = json!({"prompt": "ab", "n_predict": 1});
            for (k, v) in extra.as_object().expect("an object") {
                body[k] = v.clone();
            }
            roundtrip(addr, "POST", "/completion", &body.to_string())
        };
        let on = [
            ("typical_p", json!(0.9)),
            ("mirostat", json!(1)),
            ("mirostat", json!(2)),
            ("dry_multiplier", json!(0.8)),
            ("dry_multiplier", json!(-0.5)),
            ("xtc_probability", json!(0.5)),
            ("top_n_sigma", json!(1.0)),
            ("dynatemp_range", json!(0.5)),
        ];
        for (k, v) in on {
            let (status, text) = ask(json!({ k: v }));
            assert_eq!(status, 400, "{k} {v}: {text}");
            assert_eq!(
                message_of(&text),
                format!("{k} is not supported by this server")
            );
        }
        let chat = json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1,
            "typical_p": 0.9});
        let (status, text) = roundtrip(addr, "POST", "/v1/chat/completions", &chat.to_string());
        assert_eq!(status, 400, "{text}");
        assert_eq!(
            message_of(&text),
            "typical_p is not supported by this server"
        );
        let off = json!({
            "typical_p": 1.0, "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
            "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2,
            "dry_penalty_last_n": 64, "dry_sequence_breakers": ["\n", ":", "\"", "*"],
            "xtc_probability": 0.0, "xtc_threshold": 0.1, "top_n_sigma": -1.0,
            "dynatemp_range": 0.0, "dynatemp_exponent": 1.0,
        });
        let (status, text) = ask(off.clone());
        assert_eq!(status, 200, "{text}");
        // The settings report each of them at llama-server's off value, the
        // one this request sent, so a client reads what the server applies.
        let v: Value = serde_json::from_str(&text).expect("JSON");
        for (k, want) in off.as_object().expect("an object") {
            assert_eq!(&v["generation_settings"][k], want, "{k}");
        }
        for extra in [
            json!({"typical_p": 1.5}),
            json!({"mirostat": -1}),
            json!({"xtc_probability": -0.2}),
            json!({"top_n_sigma": 0.0}),
            json!({"dynatemp_range": -1.0}),
            json!({"mirostat_tau": 9.0, "dry_base": 0.5, "xtc_threshold": 0.9}),
            json!({"typical_p": null, "dry_sequence_breakers": null}),
        ] {
            let (status, text) = ask(extra.clone());
            assert_eq!(status, 200, "{extra}: {text}");
        }
        for (extra, want) in [
            (
                json!({"dry_allowed_length": -1}),
                format!(
                    "dry_allowed_length must be an integer in 0..={}, as llama-server takes it, not -1",
                    i32::MAX
                ),
            ),
            (
                json!({"dry_penalty_last_n": -1}),
                format!(
                    "dry_penalty_last_n must be an integer in 0..={}, as llama-server takes it, not -1",
                    i32::MAX
                ),
            ),
            (
                json!({"dry_sequence_breakers": []}),
                "dry_sequence_breakers must be a non-empty array of strings, as llama-server takes it".to_string(),
            ),
            (
                json!({"dynatemp_exponent": "1"}),
                "dynatemp_exponent must be a number, not a string".to_string(),
            ),
        ] {
            let (status, text) = ask(extra.clone());
            assert_eq!(status, 400, "{extra}: {text}");
            assert_eq!(message_of(&text), want, "{extra}");
        }
    }

    /// The error message of a 400 answer.
    fn message_of(text: &str) -> String {
        serde_json::from_str::<Value>(text).expect("the answer is JSON")["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }

    /// A field sent with a type it does not take is a 400 naming the field,
    /// the type it takes and the type it got, on every route that reads it; a
    /// `null` is unset and a value of the right type is read, never refused.
    /// The cases are the fields `gen_params`, the completion and chat
    /// halves, `/apply-template` and `/tokenize` read through the typed
    /// getters.
    #[test]
    fn a_field_of_another_type_is_a_400_naming_the_type_it_takes() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let completion = json!({"prompt": "ab", "n_predict": 1});
        let chat = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1});
        let tokenize = json!({"content": "ab"});
        let cases = [
            (
                "/completion",
                &completion,
                "stop",
                json!(5),
                "a string or an array of strings",
                "a number",
            ),
            (
                "/completion",
                &completion,
                "stop",
                json!({"a": 1}),
                "a string or an array of strings",
                "an object",
            ),
            (
                "/completion",
                &completion,
                "stream",
                json!("true"),
                "a boolean",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "stream",
                json!(1),
                "a boolean",
                "a number",
            ),
            (
                "/completion",
                &completion,
                "ignore_eos",
                json!("yes"),
                "a boolean",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "timings_per_token",
                json!(0),
                "a boolean",
                "a number",
            ),
            (
                "/completion",
                &completion,
                "return_progress",
                json!([]),
                "a boolean",
                "an array",
            ),
            (
                "/completion",
                &completion,
                "cache_prompt",
                json!("no"),
                "a boolean",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "return_tokens",
                json!(1),
                "a boolean",
                "a number",
            ),
            (
                "/completion",
                &completion,
                "n_predict",
                json!("8"),
                "an integer",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "max_tokens",
                json!(true),
                "an integer",
                "a boolean",
            ),
            (
                "/completion",
                &completion,
                "max_completion_tokens",
                json!({}),
                "an integer",
                "an object",
            ),
            (
                "/completion",
                &completion,
                "top_k",
                json!("40"),
                "an integer",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "seed",
                json!("1"),
                "an integer",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "temperature",
                json!("hot"),
                "a number",
                "a string",
            ),
            (
                "/completion",
                &completion,
                "top_p",
                json!([]),
                "a number",
                "an array",
            ),
            (
                "/completion",
                &completion,
                "min_p",
                json!(false),
                "a number",
                "a boolean",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "stop",
                json!(false),
                "a string or an array of strings",
                "a boolean",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "stream",
                json!("true"),
                "a boolean",
                "a string",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "n",
                json!("2"),
                "an integer",
                "a string",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "tools",
                json!({"type": "function"}),
                "an array",
                "an object",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "chat_template_kwargs",
                json!(5),
                "an object",
                "a number",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "reasoning_effort",
                json!(5),
                "a string",
                "a number",
            ),
            (
                "/v1/chat/completions",
                &chat,
                "add_generation_prompt",
                json!("yes"),
                "a boolean",
                "a string",
            ),
            (
                "/apply-template",
                &chat,
                "add_generation_prompt",
                json!(0),
                "a boolean",
                "a number",
            ),
            (
                "/apply-template",
                &chat,
                "return_prompt_ids",
                json!("true"),
                "a boolean",
                "a string",
            ),
            (
                "/tokenize",
                &tokenize,
                "add_special",
                json!("true"),
                "a boolean",
                "a string",
            ),
            (
                "/tokenize",
                &tokenize,
                "with_pieces",
                json!(1),
                "a boolean",
                "a number",
            ),
        ];
        for (path, base, field, value, takes, got) in cases {
            let mut body = base.clone();
            body[field] = value.clone();
            let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
            assert_eq!(status, 400, "{path} {field} {value}: {text}");
            assert_eq!(
                message_of(&text),
                format!("{field} must be {takes}, not {got}"),
                "{path} {field} {value}"
            );
            // The same field `null` is unset.
            body[field] = Value::Null;
            let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
            assert_eq!(status, 200, "{path} {field} null: {text}");
        }
        // A message content that is no string, array or `null` names its place.
        let body = json!({"messages": [{"role": "user", "content": 5}], "max_tokens": 1});
        let (status, text) = roundtrip(addr, "POST", "/v1/chat/completions", &body.to_string());
        assert_eq!(status, 400, "{text}");
        assert_eq!(
            message_of(&text),
            "messages[0].content must be a string or an array of parts, not a number"
        );
        // Values of the types the fields take are read.
        let ok = json!({
            "prompt": "ab", "n_predict": 1, "stop": ["zz"], "stream": false, "ignore_eos": false,
            "cache_prompt": true, "return_tokens": true, "top_k": 40, "seed": 7,
            "temperature": 0.5, "top_p": 0.9, "min_p": 0.1,
        });
        let (status, text) = roundtrip(addr, "POST", "/completion", &ok.to_string());
        assert_eq!(status, 200, "{text}");
    }

    /// The token limit is the first set of `n_predict`, `max_completion_tokens`
    /// and `max_tokens`, on the completion and the chat routes alike, as
    /// llama-server's task parameters read them (`n_predict` with the other two
    /// as aliases in that order); `null` is unset, and a shadowed alias of the
    /// wrong type is still a 400.
    #[test]
    fn the_token_limit_is_the_first_set_of_llama_servers_names() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let cases = [
            (
                json!({"n_predict": 2, "max_completion_tokens": 3, "max_tokens": 4}),
                2,
            ),
            (json!({"max_completion_tokens": 3, "max_tokens": 4}), 3),
            (json!({"max_tokens": 4}), 4),
            (json!({"n_predict": null, "max_tokens": 4}), 4),
            (json!({"max_completion_tokens": null, "max_tokens": 4}), 4),
            (json!({"max_tokens": 4, "max_completion_tokens": 3}), 3),
        ];
        for (limits, want) in cases {
            let mut completion = json!({"prompt": "abcabc", "temperature": 0});
            let mut chat =
                json!({"messages": [{"role": "user", "content": "abcabc"}], "temperature": 0});
            for (k, v) in limits.as_object().expect("an object") {
                completion[k] = v.clone();
                chat[k] = v.clone();
            }
            let (status, text) = roundtrip(addr, "POST", "/completion", &completion.to_string());
            assert_eq!(status, 200, "{limits}: {text}");
            let v: Value = serde_json::from_str(&text).expect("JSON");
            assert_eq!(v["generation_settings"]["n_predict"], want, "{limits}: {v}");
            assert_eq!(v["tokens_predicted"], want, "{limits}: {v}");
            let (status, text) = roundtrip(addr, "POST", "/v1/chat/completions", &chat.to_string());
            assert_eq!(status, 200, "{limits}: {text}");
            let v: Value = serde_json::from_str(&text).expect("JSON");
            assert_eq!(v["usage"]["completion_tokens"], want, "{limits}: {v}");
        }
        let shadowed = json!({"prompt": "ab", "n_predict": 2, "max_tokens": "4"});
        let (status, text) = roundtrip(addr, "POST", "/completion", &shadowed.to_string());
        assert_eq!(status, 400, "{text}");
        assert_eq!(
            message_of(&text),
            "max_tokens must be an integer, not a string"
        );
    }

    /// The token limit is an integer in `-1..=i32::MAX`, as llama-server's
    /// hard limits take `n_predict` (`-1` is no limit, `0` generates nothing);
    /// below or above is a 400 naming the name that carried it, on both routes,
    /// never a silent "no limit".
    #[test]
    fn a_token_limit_outside_minus_one_to_i32_max_is_a_400_naming_the_field() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        for (k, n) in [
            ("n_predict", -1),
            ("max_tokens", 0),
            ("max_completion_tokens", i64::from(i32::MAX)),
        ] {
            let body = json!({"prompt": "abcabc", "temperature": 0, k: n});
            let (status, text) = roundtrip(addr, "POST", "/completion", &body.to_string());
            assert_eq!(status, 200, "{k}={n}: {text}");
            let v: Value = serde_json::from_str(&text).expect("JSON");
            assert_eq!(v["generation_settings"]["n_predict"], n, "{k}={n}: {v}");
        }
        let over = i64::from(i32::MAX) + 1;
        for (k, n) in [
            ("n_predict", -2),
            ("max_completion_tokens", -3),
            ("max_tokens", i64::MIN),
            ("n_predict", over),
        ] {
            let want = format!(
                "{k} must be an integer in -1..={}, as llama-server takes it, not {n}",
                i32::MAX
            );
            let completion = json!({"prompt": "ab", k: n});
            let (status, text) = roundtrip(addr, "POST", "/completion", &completion.to_string());
            assert_eq!(status, 400, "{k}={n}: {text}");
            assert_eq!(message_of(&text), want);
            let chat = json!({"messages": [{"role": "user", "content": "ab"}], k: n});
            let (status, text) = roundtrip(addr, "POST", "/v1/chat/completions", &chat.to_string());
            assert_eq!(status, 400, "{k}={n}: {text}");
            assert_eq!(message_of(&text), want);
        }
        // The Responses route checks its own name before it becomes the chat
        // half's `max_tokens`.
        let responses = json!({"input": "ab", "max_output_tokens": -2});
        let (status, text) = roundtrip(addr, "POST", "/v1/responses", &responses.to_string());
        assert_eq!(status, 400, "{text}");
        assert_eq!(
            message_of(&text),
            format!(
                "max_output_tokens must be an integer in -1..={}, as llama-server takes it, not -2",
                i32::MAX
            )
        );
        // The range holds the name taken: a shadowed alias out of range is not
        // read, and a null one is unset, so the next name is taken.
        let shadowed = json!({"prompt": "ab", "n_predict": 1, "max_tokens": -2});
        let (status, text) = roundtrip(addr, "POST", "/completion", &shadowed.to_string());
        assert_eq!(status, 200, "{text}");
        let unset = json!({"prompt": "ab", "n_predict": null, "max_tokens": -2});
        let (status, text) = roundtrip(addr, "POST", "/completion", &unset.to_string());
        assert_eq!(status, 400, "{text}");
        assert_eq!(
            message_of(&text),
            format!(
                "max_tokens must be an integer in -1..={}, as llama-server takes it, not -2",
                i32::MAX
            )
        );
    }

    /// The address of a mock server whose chat template is `template`.
    fn served_with(template: &str) -> SocketAddr {
        super::testserve::spawn(Box::new(crate::MockEngine::new(64)), templated(template)).0
    }

    /// A body whose assistant message calls `get_weather` with `arguments`.
    fn tool_call_body(arguments: Value) -> Value {
        json!({"messages": [
            {"role": "user", "content": "w?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": arguments}},
            ]},
        ]})
    }

    /// Templates that read a tool call's `arguments` three ways: walking it as
    /// an object, printing `tojson` of it, and printing it as it came.
    const WALKS_ARGUMENTS: &str = concat!(
        "{%- for m in messages %}<{{ m.role }}>{{ m.content }}",
        "{%- for tc in m.tool_calls or [] %}[{{ tc.function.name }}",
        "{%- for k, v in tc.function.arguments.items() %} {{ k }}={{ v }}{%- endfor %}]",
        "{%- endfor %}{%- endfor %}",
    );
    const TOJSONS_ARGUMENTS: &str = concat!(
        "{%- for m in messages %}<{{ m.role }}>{{ m.content }}",
        "{%- for tc in m.tool_calls or [] %}[{{ tc.function.name }} ",
        "{{ tc.function.arguments | tojson }}]{%- endfor %}{%- endfor %}",
    );
    const PRINTS_ARGUMENTS: &str = concat!(
        "{%- for m in messages %}<{{ m.role }}>{{ m.content }}",
        "{%- for tc in m.tool_calls or [] %}[{{ tc.function.name }} ",
        "{{ tc.function.arguments }}]{%- endfor %}{%- endfor %}",
    );

    /// An assistant tool call's `arguments` is a JSON string in the OpenAI
    /// API. A template that needs it as an object (it walks it with
    /// `.items()`, or prints `tojson` of it, which a string would quote twice)
    /// gets it parsed, so the string renders as the object does; a string
    /// that is not JSON is a 400 naming the message, the call and the field. A
    /// template that takes the string as it came gets it so, whatever it
    /// holds.
    #[test]
    fn tool_call_arguments_reach_a_template_that_needs_objects_as_objects() {
        let args = json!({"city": "Seoul", "n": 3});
        for template in [WALKS_ARGUMENTS, TOJSONS_ARGUMENTS] {
            let addr = served_with(template);
            let (status, as_object) = applied(addr, &tool_call_body(args.clone()));
            assert_eq!(status, 200, "{as_object}");
            let want = applied_prompt(&as_object);
            assert!(want.contains("get_weather"), "{want}");
            let (status, as_string) =
                applied(addr, &tool_call_body(Value::String(args.to_string())));
            assert_eq!(status, 200, "{as_string}");
            assert_eq!(applied_prompt(&as_string), want, "{template}");
            for bad in ["not json", "", "{\"city\":"] {
                let (status, text) = applied(addr, &tool_call_body(json!(bad)));
                assert_eq!(status, 400, "{bad:?}: {text}");
                assert!(
                    message_of(&text).starts_with(
                        "messages[1].tool_calls[0].function.arguments is not valid JSON: "
                    ),
                    "{bad:?}: {text}"
                );
            }
        }
        let want = "<user>w?<assistant>[get_weather city=Seoul n=3]";
        let (_, text) = applied(served_with(WALKS_ARGUMENTS), &tool_call_body(args));
        assert_eq!(applied_prompt(&text), want);
        // A template that prints the string takes it as it came, valid JSON or not.
        let addr = served_with(PRINTS_ARGUMENTS);
        let (status, text) = applied(addr, &tool_call_body(json!("{\"city\":\"Seoul\"}")));
        assert_eq!(status, 200, "{text}");
        assert_eq!(
            applied_prompt(&text),
            "<user>w?<assistant>[get_weather {\"city\":\"Seoul\"}]"
        );
        let (status, text) = applied(addr, &tool_call_body(json!("not json")));
        assert_eq!(status, 200, "{text}");
    }

    /// Parsed `arguments` keep their numbers in the next prompt: a float
    /// prints as Python's `json.dumps` does, integers keep their decimal
    /// digits, a number past `f64` (`1e999`,
    /// which `arbitrary_precision` lets serde_json read) prints as it came,
    /// and reads as an infinity, not as zero or a missing number.
    #[test]
    fn tool_call_arguments_keep_their_numbers_in_the_prompt() {
        let args = json!(
            r#"{"n":1e999,"x":2.5,"y":1e2,"p":18446744073709551617,"m":-18446744073709551617}"#
        );
        let (status, text) = applied(
            served_with(TOJSONS_ARGUMENTS),
            &tool_call_body(args.clone()),
        );
        assert_eq!(status, 200, "{text}");
        assert_eq!(
            applied_prompt(&text),
            r#"<user>w?<assistant>[get_weather {"n": 1e+999, "x": 2.5, "y": 100.0, "p": 18446744073709551617, "m": -18446744073709551617}]"#
        );
        let reads = concat!(
            "{%- for m in messages %}{%- for tc in m.tool_calls or [] %}",
            "{%- for k, v in tc.function.arguments.items() %} {{ k }}",
            "{%- if v is number %}={{ v > 1 }},{{ v and 1 }}{%- endif %}",
            "{%- endfor %}{%- endfor %}{%- endfor %}",
        );
        let (status, text) = applied(served_with(reads), &tool_call_body(args.clone()));
        assert_eq!(status, 200, "{text}");
        assert_eq!(
            applied_prompt(&text),
            " n=True,1 x=True,1 y=True,1 p=True,1 m=False,1"
        );
        let negates = concat!(
            "{%- for m in messages %}{%- for tc in m.tool_calls or [] %}",
            "{%- for k, v in tc.function.arguments.items() %} {{ k }}",
            "{%- if v is number %}={{ -v }}{%- endif %}",
            "{%- endfor %}{%- endfor %}{%- endfor %}",
        );
        let (status, text) = applied(served_with(negates), &tool_call_body(args));
        assert_ne!(status, 200, "{text}");
        assert!(message_of(&text).contains("not a finite number"), "{text}");
    }

    /// Integer expressions either keep the exact value or name the existing
    /// arithmetic range limit instead of converting the operands to floats.
    #[test]
    fn tool_call_argument_integer_expressions_keep_their_values() {
        let vars = serde_json::from_str::<serde_json::Map<String, Value>>(
            r#"{"p":18446744073709551617,"q":18446744073709551616,"m":-18446744073709551617,"u":18446744073709551615,"f":18446744073709551616.0}"#,
        )
        .unwrap();
        let mut failures = Vec::new();
        for (source, want) in [
            (
                "{{ p }}|{{ m }}|{{ [p, m] }}",
                "18446744073709551617|-18446744073709551617|[18446744073709551617, -18446744073709551617]",
            ),
            (
                "{{ -p }}|{{ -m }}|{{ -u }}",
                "-18446744073709551617|18446744073709551617|-18446744073709551615",
            ),
            ("{{ p == q }}|{{ p > q }}|{{ m < -q }}", "False|True|True"),
            (
                "{{ p == f }}|{{ p > f }}|{{ q == f }}|{{ f < p }}",
                "False|True|True|True",
            ),
            ("{{ [p] == [q] }}|{{ p in [q] }}", "False|False"),
            ("{{ p and 1 }}|{{ m and 1 }}|{{ u and 1 }}", "1|1|1"),
        ] {
            let got = ChatTemplate::parse(source).unwrap().render(&vars);
            if !got.as_deref().is_ok_and(|text| text == want) {
                failures.push(format!("{source}: {got:?}, expected {want}"));
            }
        }
        for source in [
            "{{ p + 1 }}",
            "{{ p - 1 }}",
            "{{ p * 2 }}",
            "{{ p // 2 }}",
            "{{ p % 2 }}",
        ] {
            let got = ChatTemplate::parse(source).unwrap().render(&vars);
            if !got
                .as_ref()
                .is_err_and(|e| e.0.contains("integer overflow"))
            {
                failures.push(format!("{source}: {got:?}, expected integer overflow"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    use super::fixture_templates::FIXTURE_TEMPLATES;

    /// Which of the fixture chat templates need a tool call's `arguments` as
    /// an object ([`needs_object_arguments`]): the ones that walk it
    /// (Qwen3.6's, Qwen3.8's, Nemotron's and Bolt's `|items`, GLM's
    /// `.items()`) do, and the ones that take a string (Qwen3's `is string`
    /// branch, V4.1's `from_json`) do not, nor does a template that prints no
    /// arguments. The mock's inline template prints none.
    #[test]
    fn the_fixture_templates_that_need_object_arguments_are_named() {
        let parsed = |source: &str| ChatTemplate::parse(source).expect("a template");
        let want = |name: &str| match name {
            "qwen3" | "v41" => false,
            "qwen36" | "qwen38" | "glm5" | "nemotron" | "nemotron-bolt" => true,
            other => panic!("{other} is not in the object-arguments table"),
        };
        for (name, source) in FIXTURE_TEMPLATES {
            assert_eq!(
                needs_object_arguments(&parsed(source)),
                want(name),
                "{name}"
            );
        }
        assert!(
            !needs_object_arguments(&parsed(super::testserve::TEMPLATE)),
            "mock"
        );
    }

    /// How each fixture chat template answers `reasoning_effort: "none"`
    /// ([`probe_thinking_off`]), read from its renders: Qwen3's, Qwen3.6's,
    /// Qwen3.8's, V4.1's, Nemotron's and Bolt's generation prompts change with
    /// `enable_thinking` ([`ThinkingOff::Switch`]); GLM's always opens
    /// `<think>` and changes only its effort header
    /// ([`ThinkingOff::LowEffort`]); the mock's reads neither
    /// ([`ThinkingOff::Neither`]). Only the second prints a line at bind.
    #[test]
    fn the_fixture_templates_name_how_they_turn_thinking_off() {
        let parsed = |source: &str| ChatTemplate::parse(source).expect("a template");
        let want = |name: &str| match name {
            "qwen3" | "qwen36" | "qwen38" | "nemotron" | "nemotron-bolt" => {
                ThinkingOff::Switch { opt_in: false }
            }
            "v41" => ThinkingOff::Switch { opt_in: true },
            "glm5" => ThinkingOff::LowEffort,
            other => panic!("{other} is not in the thinking-off table"),
        };
        for (name, source) in FIXTURE_TEMPLATES {
            let got = probe_thinking_off(&parsed(source)).expect("the probe renders");
            assert_eq!(got, want(name), "{name}");
            assert_eq!(
                got.note().is_some(),
                want(name) == ThinkingOff::LowEffort,
                "{name}: the bind line"
            );
        }
        let got =
            probe_thinking_off(&parsed(super::testserve::TEMPLATE)).expect("the probe renders");
        assert_eq!(got, ThinkingOff::Neither, "mock");
        assert!(!got.note().is_some(), "mock: the bind line");
    }

    /// A template that cannot render the probe's one-message conversation is
    /// the bind's error naming the probe, never a guess: the probe and
    /// [`Server::bind`] both end in [`ServeError::Template`].
    #[test]
    fn a_template_the_thinking_probe_cannot_render_refuses_the_bind() {
        let refuses = "{{- raise_exception('nothing renders') }}";
        let e = probe_thinking_off(&ChatTemplate::parse(refuses).expect("a template"))
            .expect_err("the probe fails");
        assert!(
            e.to_string().contains("thinking probe") && e.to_string().contains("nothing renders"),
            "{e}"
        );
        match super::Server::bind(
            "127.0.0.1:0",
            Box::new(crate::MockEngine::new(64)),
            templated(refuses),
        ) {
            Err(ServeError::Template(e)) => {
                assert!(e.to_string().contains("thinking probe"), "{e}");
            }
            Err(other) => panic!("not a template error: {other}"),
            Ok(_) => panic!("the bind served a template nothing renders"),
        }
    }

    /// `samplers` is taken only in this server's order, by llama-server's
    /// names, their aliases or its characters, with every sampler that is off
    /// at the request's values dropped from both lists (so llama-server's
    /// default list passes, and an off stage may stand anywhere); another
    /// order of the stages that are on, an unknown name or character, or a
    /// sampler this server lacks that is on is a 400, on every generation
    /// route.
    #[test]
    fn samplers_are_taken_only_in_this_servers_order() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let ask = |path: &str, samplers: &Value| {
            let mut body = if path == "/completion" || path == "/v1/completions" {
                json!({"prompt": "ab", "max_tokens": 1})
            } else {
                json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1})
            };
            // A penalty turns the penalties stage on, so its place counts.
            body["repeat_penalty"] = json!(1.1);
            body["samplers"] = samplers.clone();
            roundtrip(addr, "POST", path, &body.to_string())
        };
        for path in ["/completion", "/v1/completions", "/v1/chat/completions"] {
            for ok in [
                json!([
                    "penalties",
                    "dry",
                    "top_n_sigma",
                    "top_k",
                    "typ_p",
                    "top_p",
                    "min_p",
                    "xtc",
                    "temperature"
                ]),
                json!("edskypmxt"),
                json!(["penalties", "top_k", "top_p", "min_p", "temperature"]),
                json!(["Penalties", "top-k", "nucleus", "minp", "temp"]),
                json!("ekpmt"),
                Value::Null,
            ] {
                let (status, text) = ask(path, &ok);
                assert_eq!(status, 200, "{path} {ok}: {text}");
            }
            for bad in [
                json!(["dry", "top_k", "penalties", "top_p", "min_p", "temperature"]),
                json!(["top_k", "penalties", "top_p", "min_p", "temperature"]),
                json!([
                    "penalties",
                    "top_k",
                    "top_p",
                    "min_p",
                    "temperature",
                    "adaptive_p"
                ]),
                json!(["penalties", "top_k", "top_p", "min_p", "warmth"]),
                json!([]),
                json!("kpmte"),
                json!("ekpmtz"),
                json!(["penalties", 1, "top_p", "min_p", "temperature"]),
                json!([
                    "penalties",
                    "penalties",
                    "top_k",
                    "top_p",
                    "min_p",
                    "temperature"
                ]),
                json!(["penalties", "top_\u{212A}", "top_p", "min_p", "temperature"]),
                json!("EKPMT"),
            ] {
                let (status, text) = ask(path, &bad);
                assert_eq!(status, 400, "{path} {bad}: {text}");
                assert!(
                    message_of(&text).starts_with("samplers must be this server's order"),
                    "{path} {bad}: {text}"
                );
            }
            let mut on = if path == "/v1/chat/completions" {
                json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1})
            } else {
                json!({"prompt": "ab", "max_tokens": 1})
            };
            on["samplers"] = json!("edskypmxt");
            for (field, value) in [
                ("typical_p", json!(0.9)),
                ("dry_multiplier", json!(0.5)),
                ("xtc_probability", json!(0.5)),
                ("top_n_sigma", json!(1.0)),
            ] {
                let mut body = on.clone();
                body[field] = value;
                let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
                assert_eq!(status, 400, "{path} {field}: {text}");
                assert_eq!(
                    message_of(&text),
                    format!("{field} is not supported by this server")
                );
            }
            // A stage at its off value drops out of the comparison: with no
            // penalty the penalties stage may stand anywhere, top_k 0 may be
            // left out, and a greedy request orders none of the later stages.
            let plain = if path == "/v1/chat/completions" {
                json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 1})
            } else {
                json!({"prompt": "ab", "max_tokens": 1})
            };
            for (extra, samplers) in [
                (
                    json!({}),
                    json!(["top_k", "top_p", "penalties", "min_p", "temperature"]),
                ),
                (
                    json!({"top_k": 0}),
                    json!(["penalties", "top_p", "min_p", "temperature"]),
                ),
                (
                    json!({"temperature": 0}),
                    json!(["temperature", "min_p", "top_k"]),
                ),
                (
                    json!({"temperature": 1}),
                    json!(["penalties", "top_k", "top_p", "min_p"]),
                ),
                (
                    json!({"temperature": 0, "repeat_penalty": 1.1}),
                    json!(["penalties", "top_k", "top_p", "min_p", "temperature"]),
                ),
            ] {
                let mut body = plain.clone();
                for (k, v) in extra.as_object().expect("an object") {
                    body[k] = v.clone();
                }
                body["samplers"] = samplers.clone();
                let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
                assert_eq!(status, 200, "{path} {extra} {samplers}: {text}");
            }
            // Temperature is never off, and a penalty before a greedy pick makes
            // the stages around it count again.
            for (extra, samplers) in [
                (
                    json!({"top_k": 0}),
                    json!(["penalties", "top_p", "temperature"]),
                ),
                (json!({"temperature": 0}), json!(["min_p", "top_k"])),
                (json!({"temperature": 0}), json!([])),
                (
                    json!({"temperature": 0, "repeat_penalty": 1.1}),
                    json!(["temperature", "penalties"]),
                ),
                (
                    json!({"temperature": 0, "repeat_penalty": 1.1}),
                    json!(["top_p", "penalties", "temperature"]),
                ),
            ] {
                let mut body = plain.clone();
                for (k, v) in extra.as_object().expect("an object") {
                    body[k] = v.clone();
                }
                body["samplers"] = samplers.clone();
                let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
                assert_eq!(status, 400, "{path} {extra} {samplers}: {text}");
            }
            let (status, text) = ask(path, &json!(5));
            assert_eq!(status, 400, "{text}");
            assert_eq!(
                message_of(&text),
                "samplers must be an array of sampler names or a string of sampler characters, \
                 not a number"
            );
        }
    }

    /// `/props` names how the template turns thinking off, in the words of
    /// [`ThinkingOff::word`].
    #[test]
    fn props_names_how_the_template_turns_thinking_off() {
        let cases = [
            (
                include_str!("../tests/fixtures/qwen38-chat-template.jinja"),
                "switch",
            ),
            (
                include_str!("../tests/fixtures/glm5-chat-template.jinja"),
                "low_effort",
            ),
            (super::testserve::TEMPLATE, "neither"),
        ];
        for (template, want) in cases {
            let (addr, _state, _ended) =
                super::testserve::spawn(Box::new(crate::MockEngine::new(64)), templated(template));
            let (status, body) = roundtrip(addr, "GET", "/props", "");
            assert_eq!(status, 200, "{body}");
            let props: Value = serde_json::from_str(&body).expect("/props is JSON");
            assert_eq!(props["thinking_off"], want, "{body}");
        }
    }

    /// A sampler factory that refuses a request's parameters ends that request
    /// with a 400 carrying the refusal, on every route and stream or not, before
    /// the request takes a slot: a greedy request builds no sampler and runs, a
    /// sampled one that is not refused runs, and the one slot is free after
    /// each.
    #[test]
    fn a_refusing_sampler_factory_is_the_requests_400() {
        let inner = crate::sampling::reference_factory();
        let asked = Arc::new(AtomicUsize::new(0));
        let mut config = super::testserve::mock_config();
        config.sampler = Some({
            let asked = Arc::clone(&asked);
            Arc::new(move |p: &SamplingParams| {
                asked.fetch_add(1, Ordering::SeqCst);
                if p.top_k == 7 {
                    return Err(SamplerRefused("top_k 7 is not takeable".to_owned()));
                }
                inner(p)
            })
        });
        let (addr, _state, _ended) =
            super::testserve::spawn(Box::new(crate::MockEngine::new(64)), config);
        let completion = |extra: Value| {
            let mut b = json!({"prompt": "ab", "n_predict": 1});
            b.as_object_mut()
                .expect("an object")
                .extend(extra.as_object().expect("an object").clone());
            b
        };
        let chat = |extra: Value| {
            let mut b = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1});
            b.as_object_mut()
                .expect("an object")
                .extend(extra.as_object().expect("an object").clone());
            b
        };
        let refusal = "the sampler refuses the sampling parameters: top_k 7 is not takeable";
        for (path, body) in [
            ("/completion", completion(json!({"top_k": 7}))),
            (
                "/completion",
                completion(json!({"top_k": 7, "stream": true})),
            ),
            ("/v1/chat/completions", chat(json!({"top_k": 7}))),
            (
                "/v1/chat/completions",
                chat(json!({"top_k": 7, "stream": true})),
            ),
            ("/v1/completions", completion(json!({"top_k": 7}))),
        ] {
            let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
            assert_eq!(status, 400, "{path} {body}: {text}");
            assert_eq!(message_of(&text), refusal, "{path} {body}");
        }
        assert_eq!(
            asked.load(Ordering::SeqCst),
            5,
            "each refused request asked once"
        );
        // Greedy: no sampler is built, so the factory is not asked.
        let (status, text) = roundtrip(
            addr,
            "POST",
            "/completion",
            &completion(json!({"top_k": 7, "temperature": 0})).to_string(),
        );
        assert_eq!(status, 200, "{text}");
        assert_eq!(
            asked.load(Ordering::SeqCst),
            5,
            "a greedy request asks nothing"
        );
        let (status, text) = roundtrip(
            addr,
            "POST",
            "/completion",
            &completion(json!({})).to_string(),
        );
        assert_eq!(status, 200, "{text}");
        assert_eq!(asked.load(Ordering::SeqCst), 6);
        let (status, text) = roundtrip(addr, "GET", "/health", "");
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(v["slots_idle"], 1, "{v}");
    }

    /// A round of two slots whose rows split into one step and one drafted
    /// pass — a request that asks for probabilities steps beside a greedy
    /// one that passes — runs the pass as the engine's
    /// `advance`, the call a request alone makes, not as a round of one row
    /// ([`crate::Engine::advance_slots`] is called with two rows or more). The
    /// long greedy request holds the engine thread in its prompt call until
    /// the short stepping one waits behind it, so the two share rounds, which
    /// the call log shows as a pass, a step and a pass in a row.
    #[test]
    fn a_mixed_rounds_single_pass_is_an_advance() {
        use crate::mock::{DraftMock, Hold, MockCall};
        use std::sync::Mutex;
        let log = Arc::new(Mutex::new(Vec::new()));
        let hold = Arc::new(Hold::default());
        let mock = DraftMock::new(4096)
            .with_slots(2)
            .logged(Arc::clone(&log))
            .holding(Arc::clone(&hold));
        let server = super::Server::bind_with(
            "127.0.0.1:0",
            Box::new(mock),
            super::testserve::mock_config(),
            SlotConfig {
                parallel: 2,
                ..SlotConfig::default()
            },
        )
        .expect("bind");
        let addr = server.local_addr().expect("addr");
        let (_state, _ended) = server.start().expect("start");
        // Each request is written and its answer read once the engine thread
        // is free, so the two share rounds without a thread of the test's own.
        let ask = |body: Value| {
            super::testserve::send_request(addr, "POST", "/completion", &[], &body.to_string())
        };
        let long = ask(
            json!({"prompt": "abcabcabc", "n_predict": 60, "temperature": 0,
                   "cache_prompt": false}),
        );
        assert!(
            hold.reached(1, Duration::from_secs(10)),
            "the long prompt never reached the engine"
        );
        let short = ask(
            json!({"prompt": "abacadae", "n_predict": 12, "temperature": 0,
                   "n_probs": 1, "cache_prompt": false}),
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while !roundtrip(addr, "GET", "/metrics", "")
            .1
            .contains("llamacpp:requests_deferred 1")
        {
            assert!(Instant::now() < deadline, "the short request never queued");
            std::thread::sleep(Duration::from_millis(20));
        }
        hold.open();
        for (name, mut stream) in [("long", long), ("short", short)] {
            let mut answer = String::new();
            stream.read_to_string(&mut answer).expect(name);
            assert_eq!(
                answer.split_whitespace().nth(1),
                Some("200"),
                "{name}: {answer}"
            );
        }
        let calls = log.lock().expect("the call log").clone();
        assert!(
            !calls.contains(&MockCall::AdvanceSlots(1)),
            "a pass of one row went through advance_slots: {calls:?}"
        );
        assert!(
            calls
                .windows(3)
                .any(|w| w == [MockCall::Advance, MockCall::Next, MockCall::Advance]),
            "no round carried a step and a pass: {calls:?}"
        );
    }

    // ---------------------------------------------------------------- API keys

    /// The key every key test sets: known bytes, so a leak is a named string.
    const KEY: &str = "sk-this-is-the-secret-key";
    /// `KEY` as the `Authorization` header carries it; the assert pins the
    /// two spellings together.
    const BEARER: &str = "Bearer sk-this-is-the-secret-key";

    fn auth(v: &'static str) -> [(&'static str, &'static str); 1] {
        [("Authorization", v)]
    }

    fn xkey(v: &'static str) -> [(&'static str, &'static str); 1] {
        [("X-Api-Key", v)]
    }

    /// The key check over a real socket, as llama-server's security tests
    /// drive it: the public paths answer without a key; a generation request
    /// without one, or with a wrong one — `Bearer`-prefixed or bare — is
    /// llama-server's 401 (`Invalid API Key`, `authentication_error`, the
    /// JSON content type); a valid key answers 200 from either header, with
    /// or without the `Bearer ` prefix; `OPTIONS` passes with no key at all,
    /// as llama-server's pre-routing handler answers it before the check, and
    /// allows every method a route takes, `DELETE` included.
    #[test]
    fn the_key_check_answers_llama_servers_401_and_lets_the_keyed_through() {
        assert_eq!(BEARER, format!("Bearer {KEY}"));
        let mut keys = ApiKeys::default();
        keys.add("--api-key", KEY).unwrap();
        let (addr, _state, _ended) = super::testserve::spawn_keyed(keys);

        for path in ["/health", "/v1/health"] {
            let (status, _head, body) =
                super::testserve::roundtrip_head(addr, "GET", path, &[], "");
            assert_eq!(status, 200, "{path}: {body}");
        }
        let (status, head, _body) =
            super::testserve::roundtrip_head(addr, "OPTIONS", "/completion", &[], "");
        assert_eq!(status, 204, "OPTIONS passes with no key: {head}");
        assert!(
            head.contains("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS"),
            "the preflight allows every routed method: {head}"
        );

        let ask = r#"{"prompt":"I believe the meaning of life is","max_tokens":4}"#;
        let (status, head, body) =
            super::testserve::roundtrip_head(addr, "POST", "/completion", &[], ask);
        assert_eq!(status, 401, "no key: {body}");
        assert!(
            head.contains("Content-Type: application/json; charset=utf-8"),
            "llama-server's content type: {head}"
        );
        let v: Value = serde_json::from_str(&body).expect("the 401's body is JSON");
        assert_eq!(v["error"]["type"], "authentication_error");
        assert_eq!(v["error"]["message"], "Invalid API Key");
        assert_eq!(v["error"]["code"], 401);

        for wrong in ["Bearer sk-wrong", "sk-wrong", ""] {
            let (status, body) =
                super::testserve::roundtrip(addr, "POST", "/completion", &auth(wrong), ask);
            assert_eq!(status, 401, "Authorization {wrong:?}: {body}");
        }
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/completion", &xkey("sk-wrong"), ask);
        assert_eq!(status, 401, "X-Api-Key wrong: {body}");

        for right in [BEARER, KEY] {
            let (status, body) =
                super::testserve::roundtrip(addr, "POST", "/completion", &auth(right), ask);
            assert_eq!(status, 200, "Authorization {right:?}: {body}");
        }
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/completion", &xkey(KEY), ask);
        assert_eq!(status, 200, "X-Api-Key: {body}");
        assert!(body.contains("content"), "the generation answered: {body}");
    }

    /// A key file line's scope and limits over a real socket: a key whose
    /// `scope=` does not reach the route is a 403 (`permission_error`) and a
    /// route it reaches answers; `inflight=1` refuses a second request with a
    /// 429 and `Retry-After: 1` while a background response the key started
    /// still generates (a call on the stored response passes), and admits
    /// again once it ends, and at once after each answer; `rate=1/m` refuses
    /// the second request with the remaining wait rounded up in `Retry-After` on
    /// the same keep-alive connection, which carries the next request after it.
    #[test]
    fn a_key_files_scope_and_limits_answer_403_and_429() {
        use crate::mock::{DraftMock, Hold};
        use std::io::{BufReader, Write};
        use std::sync::Arc;
        let (keys, _) = crate::flag::key_file(
            "admit",
            "sk-gen scope=inference inflight=1\nsk-look scope=read\nsk-slow rate=1/m\n",
        );
        let hold = Arc::new(Hold::default());
        let mut config = super::testserve::mock_config();
        config.api_keys = keys.expect("the key file");
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(DraftMock::new(4096).holding(Arc::clone(&hold))),
            config,
        );
        // Each answer is bounded: a request the limits should refuse would
        // otherwise wait for the held engine.
        let call = |key: &str, method: &str, path: &str, body: &str| {
            let bearer = format!("Bearer {key}");
            let auth = [("Authorization", bearer.as_str())];
            let mut s = super::testserve::send_request(addr, method, path, &auth, body);
            s.set_read_timeout(Some(Duration::from_secs(10)))
                .expect("timeout");
            let mut read = BufReader::new(&mut s);
            let (status, head, body) = crate::http::read_test_response(&mut read);
            assert_eq!(read.read(&mut [0]).expect("the connection closes"), 0);
            (
                status,
                head,
                String::from_utf8(body).expect("a UTF-8 response"),
            )
        };
        let ask = r#"{"prompt":"abab","n_predict":2}"#;

        let (status, head, body) = call("sk-look", "POST", "/completion", ask);
        assert_eq!(status, 403, "{body}");
        assert!(head.starts_with("HTTP/1.0 403 Forbidden\r\n"), "{head}");
        let v: Value = serde_json::from_str(&body).expect("the 403's JSON");
        assert_eq!(v["error"]["type"], "permission_error");
        assert_eq!(
            v["error"]["message"],
            "this API key's scope does not reach POST /completion"
        );
        assert_eq!(call("sk-look", "GET", "/props", "").0, 200);
        assert_eq!(call("sk-gen", "GET", "/props", "").0, 403);

        let background = r#"{"input":"abab","background":true,"max_output_tokens":2}"#;
        let (status, _, body) = call("sk-gen", "POST", "/v1/responses", background);
        assert_eq!(status, 200, "{body}");
        let id = serde_json::from_str::<Value>(&body).unwrap()["id"]
            .as_str()
            .expect("the response's id")
            .to_owned();
        assert!(
            hold.reached(1, Duration::from_secs(10)),
            "the background generation never reached the engine"
        );
        let (status, head, body) = call("sk-gen", "POST", "/completion", ask);
        assert_eq!(status, 429, "{body}");
        assert!(
            head.starts_with("HTTP/1.0 429 Too Many Requests\r\n"),
            "{head}"
        );
        assert!(head.contains("Retry-After: 1\r\n"), "{head}");
        let v: Value = serde_json::from_str(&body).expect("the 429's JSON");
        assert_eq!(v["error"]["type"], "rate_limit_error");
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("in-flight limit (inflight=1)")),
            "{body}"
        );
        assert_eq!(
            call("sk-slow", "GET", "/props", "").0,
            200,
            "another key's room"
        );
        // A stored-response call runs no generation: it takes no place, so
        // the key can read (or cancel) the response that holds its one.
        let stored = format!("/v1/responses/{id}");
        let (status, _, body) = call("sk-gen", "GET", &stored, "");
        assert_eq!(status, 200, "{body}");
        assert!(super::responses::store_call(
            "POST",
            &format!("{stored}/cancel")
        ));
        assert!(super::responses::store_call("DELETE", &stored));
        assert!(!super::responses::store_call(
            "POST",
            "/v1/responses/input_tokens"
        ));
        hold.open();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let (status, _, body) = call("sk-gen", "GET", &stored, "");
            if status == 200 && body.contains(r#""status":"completed""#) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the background response never ended: {body}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(call("sk-gen", "POST", "/completion", ask).0, 200);
        // The place is given back once the answer is written, before its
        // connection closes, so a client that asks again at once is never
        // refused.
        for _ in 0..50 {
            let (status, _, body) = call("sk-gen", "POST", "/tokenize", r#"{"content":"ab"}"#);
            assert_eq!(status, 200, "{body}");
        }

        let mut client = std::net::TcpStream::connect(addr).expect("connect");
        let mut read = BufReader::new(client.try_clone().expect("clone"));
        let mut next = || {
            write!(
                client,
                "POST /tokenize HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer sk-slow\r\n\
                 Content-Length: 16\r\n\r\n{{\"content\":\"ab\"}}"
            )
            .expect("write");
            crate::http::read_test_response(&mut read)
        };
        let (status, head, _) = next();
        assert_eq!(status, 429, "the /props above spent sk-slow's one: {head}");
        // PIN(2026-10-30): wall-clock requests accept elapsed time; admit's unit test pins exact rounding.
        assert!(
            (1..=60).any(|s| head.contains(&format!("Retry-After: {s}\r\n"))),
            "{head}"
        );
        assert!(!head.contains("Connection: close"), "{head}");
        assert_eq!(next().0, 429, "the connection carries the next request");
    }

    /// An answer holds its key's in-flight place while its write is blocked:
    /// with `inflight=1`, a client that reads a large answer's head and then
    /// stops reading keeps the key's one place, so another request of the
    /// key is a 429 until that connection ends.
    #[test]
    fn a_blocked_answer_write_keeps_its_place() {
        use crate::mock::{DraftMock, Hold};
        use std::io::{BufRead, BufReader, Write};
        use std::sync::Arc;

        for background in [false, true] {
            let (keys, _) = crate::flag::key_file("blocked", "sk-one inflight=1\n");
            let mut config = super::testserve::mock_config();
            config.api_keys = keys.expect("the key file");
            let hold = Arc::new(Hold::default());
            let (addr, state, _ended) = super::testserve::spawn(
                Box::new(DraftMock::new(64).holding(Arc::clone(&hold))),
                config,
            );
            let auth = [("Authorization", "Bearer sk-one")];
            let small = r#"{"content":"ab"}"#;
            // Both answers exceed the socket buffers, so a reader that
            // stops after the head leaves the writer blocked.
            let (path, large) = if background {
                (
                    "/v1/responses",
                    json!({"input": "abab", "model": "m".repeat(8 << 20),
                        "background": true, "max_output_tokens": 2})
                    .to_string(),
                )
            } else {
                (
                    "/tokenize",
                    json!({"content": "a".repeat(1 << 20), "with_pieces": true}).to_string(),
                )
            };
            let mut big = std::net::TcpStream::connect(addr).expect("connect");
            big.set_read_timeout(Some(Duration::from_secs(15)))
                .expect("timeout");
            write!(
                big,
                "POST {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer sk-one\r\n\
                 Content-Length: {}\r\n\r\n{large}",
                large.len()
            )
            .expect("write");
            let mut read = BufReader::new(big);
            let mut status_line = String::new();
            read.read_line(&mut status_line)
                .expect("the large answer's head");
            assert!(status_line.starts_with("HTTP/1.1 200"), "{status_line}");
            if background {
                loop {
                    let mut line = String::new();
                    assert_ne!(read.read_line(&mut line).expect("queued response head"), 0);
                    if line == "\r\n" {
                        break;
                    }
                }
                // The queued object's first field is its id. Read just
                // that field, leaving the large model string unread.
                let mut first = Vec::new();
                read.read_until(b',', &mut first).expect("the response id");
                assert_eq!(first.pop(), Some(b','));
                first.push(b'}');
                let first: Value = serde_json::from_slice(&first).expect("the id field");
                let id = first["id"].as_str().expect("a response id");
                assert!(hold.reached(1, Duration::from_secs(10)));
                hold.open();
                let deadline = Instant::now() + Duration::from_secs(15);
                loop {
                    let stored = state.responses.get(id).expect("the stored response");
                    if matches!(stored["status"].as_str(), Some("completed" | "incomplete")) {
                        break;
                    }
                    assert!(
                        Instant::now() < deadline,
                        "the background generation never ended"
                    );
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            let (status, head, body) =
                super::testserve::roundtrip_head(addr, "POST", "/tokenize", &auth, small);
            assert_eq!(
                status, 429,
                "the blocked write holds the place (background={background}): {head}{body}"
            );
            drop(read);
            let deadline = Instant::now() + Duration::from_secs(15);
            while super::testserve::roundtrip(addr, "POST", "/tokenize", &auth, small).0 != 200 {
                assert!(Instant::now() < deadline, "the place never came back");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    /// Sends `head`, then a body in pieces the server reads after it answered from the head,
    /// and returns the answer's status once the connection ends with EOF: an answer that
    /// closes with request bytes unread still reaches the client.
    fn answer_before_the_body(addr: SocketAddr, head: &str) -> u16 {
        use std::io::{BufReader, Write};
        use std::net::{Shutdown, TcpStream};

        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream.write_all(head.as_bytes()).unwrap();
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(20));
            stream
                .write_all(&[b'x'; 256])
                .expect("the server reads the rest of the request");
        }
        stream.shutdown(Shutdown::Write).unwrap();
        let mut reader = BufReader::new(stream);
        let (status, _, _) = crate::http::read_test_response(&mut reader);
        let mut byte = [0];
        assert_eq!(
            reader
                .read(&mut byte)
                .expect("the connection ends with EOF"),
            0,
            "{head:?}"
        );
        status
    }

    /// Refusals and public bodyless handlers answer headers without `100 Continue` or a body;
    /// authorized requests retain both body framings, the handshake and the next keyed request.
    #[test]
    fn authorization_precedes_bodies_and_preserves_authorized_keep_alive() {
        use crate::http::read_test_response as response;
        use std::io::{BufReader, Write};
        use std::net::TcpStream;

        fn connect(addr: SocketAddr) -> BufReader<TcpStream> {
            let stream = TcpStream::connect(addr).expect("connect");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            BufReader::new(stream)
        }

        let mut keys = ApiKeys::default();
        keys.add("--api-key", KEY).unwrap();
        let (addr, _state, _ended) = super::testserve::spawn_keyed(keys);
        for key in [
            "",
            "Authorization: Bearer sk-wrong\r\n",
            "X-Api-Key: sk-wrong\r\n",
        ] {
            for framing in ["Content-Length: 1024\r\n", "Transfer-Encoding: chunked\r\n"] {
                for expect in ["", "Expect: 100-continue\r\n"] {
                    let mut r = connect(addr);
                    write!(
                        r.get_mut(),
                        "POST /completion HTTP/1.1\r\nConnection: keep-alive\r\n\
                         {key}{framing}{expect}\r\n"
                    )
                    .expect("headers without a body");
                    let (status, head, body) = response(&mut r);
                    assert_eq!(status, 401, "{head}");
                    assert!(head.contains("Connection: close"), "{head}");
                    let v: Value = serde_json::from_slice(&body).expect("the 401's JSON");
                    assert_eq!(v["error"]["type"], "authentication_error");
                    let mut byte = [0];
                    assert_eq!(r.read(&mut byte).expect("the denied connection closes"), 0);
                }
            }
        }

        for (method, path, framing, status) in [
            ("GET", "/health", "Content-Length: 1024\r\n", 200),
            ("GET", "/v1/health", "Transfer-Encoding: chunked\r\n", 200),
            (
                "OPTIONS",
                "/completion",
                "Transfer-Encoding: chunked\r\n",
                204,
            ),
        ] {
            let mut r = connect(addr);
            write!(
                r.get_mut(),
                "{method} {path} HTTP/1.1\r\nConnection: keep-alive\r\n\
                 {framing}Expect: 100-continue\r\n\r\n"
            )
            .expect("public headers without the ignored body");
            let (got, head, _) = response(&mut r);
            assert_eq!(got, status, "{head}");
            assert!(head.contains("Connection: close"), "{head}");
            let mut byte = [0];
            assert_eq!(r.read(&mut byte).expect("the unread public body closes"), 0);
        }
        let denied = "POST /completion HTTP/1.1\r\nContent-Length: 1024\r\n\r\n";
        assert_eq!(answer_before_the_body(addr, denied), 401);

        let ask = r#"{"prompt":"hello","max_tokens":1}"#;
        for chunked in [false, true] {
            let mut r = connect(addr);
            let framing = if chunked {
                "Transfer-Encoding: chunked\r\n".to_owned()
            } else {
                format!("Content-Length: {}\r\n", ask.len())
            };
            write!(
                r.get_mut(),
                "POST /completion HTTP/1.1\r\nAuthorization: {BEARER}\r\n\
                 {framing}Expect: 100-continue\r\n\r\n"
            )
            .expect("authorized headers");
            let (status, _, body) = response(&mut r);
            assert_eq!(status, 100, "continue before sending the body");
            assert!(body.is_empty());
            if chunked {
                write!(r.get_mut(), "{:x}\r\n{ask}\r\n0\r\n\r\n", ask.len()).unwrap();
            } else {
                r.get_mut().write_all(ask.as_bytes()).unwrap();
            }
            let (status, head, body) = response(&mut r);
            assert_eq!(status, 200, "{head}");
            assert!(head.contains("Connection: keep-alive"), "{head}");
            let v: Value = serde_json::from_slice(&body).expect("completion JSON");
            assert!(v.get("content").is_some(), "{v}");

            write!(
                r.get_mut(),
                "GET /props HTTP/1.1\r\nAuthorization: {BEARER}\r\nConnection: close\r\n\r\n"
            )
            .expect("the next keyed request");
            let (status, head, body) = response(&mut r);
            assert_eq!(status, 200, "{head}");
            let v: Value = serde_json::from_slice(&body).expect("props JSON");
            assert!(v.is_object(), "{v}");
            let mut byte = [0];
            assert_eq!(r.read(&mut byte).expect("the final connection closes"), 0);
        }
    }

    /// Ambiguous lengths and broken chunk boundaries are refused before another request can run.
    #[test]
    fn malformed_request_framing_closes_the_connection() {
        use crate::http::read_test_response as response;
        use std::io::{BufReader, Write};
        use std::net::{Shutdown, TcpStream};

        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let mut accepted = Vec::new();
        let next = "GET /health HTTP/1.1\r\nConnection: close\r\n\r\n";
        let smuggled = format!("Content-Length: 0\r\ncontent-length: {}\r\n", next.len());
        let body = r#"{"content":"x"}"#;
        let whitespace_frames = ["\u{b}", "\u{c}", "\u{85}", "\u{a0}", "\u{2003}", "\r"]
            .into_iter()
            .flat_map(|space| {
                let n = body.len();
                [
                    (
                        "POST /tokenize",
                        format!("Content-Length: {space}{n}\r\n"),
                        format!("{body}{next}"),
                    ),
                    (
                        "POST /tokenize",
                        format!("Content-Length: {n}{space}\r\n"),
                        format!("{body}{next}"),
                    ),
                    (
                        "POST /tokenize",
                        format!("Content-Length: {n},{space}{n}\r\n"),
                        format!("{body}{next}"),
                    ),
                    (
                        "POST /tokenize",
                        "Transfer-Encoding: chunked\r\n".to_owned(),
                        format!("{n:x}{space}\r\n{body}\r\n0\r\n\r\n{next}"),
                    ),
                    (
                        "POST /tokenize",
                        "Transfer-Encoding: chunked\r\n".to_owned(),
                        format!("{n:x}{space};x=y\r\n{body}\r\n0\r\n\r\n{next}"),
                    ),
                    (
                        "POST /tokenize",
                        "Transfer-Encoding: chunked\r\n".to_owned(),
                        format!("{n:x}\r\n{body}\r\n0{space}\r\n\r\n{next}"),
                    ),
                    (
                        "GET /health",
                        format!("Content-Length: {space}0\r\n"),
                        next.to_owned(),
                    ),
                    (
                        "GET /v1/health",
                        format!("Content-Length: 0,{space}0\r\n"),
                        next.to_owned(),
                    ),
                    (
                        "OPTIONS /tokenize",
                        format!("Content-Length: 0{space}\r\n"),
                        next.to_owned(),
                    ),
                ]
            });
        for (start, headers, body) in [
            (
                "POST /tokenize",
                "Content-Length: 2\r\ncontent-length: 3\r\n",
                "{}",
            ),
            ("POST /tokenize", "Content-Length: 2, 3\r\n", "{}"),
            ("POST /tokenize", "Content-Length: 3\r\n", "{}"),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "2\r\n{}JUNK\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "2\r\n{}\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "2\r\n{}XX0\r\n\r\n",
            ),
            ("POST /tokenize", "Transfer-Encoding: chunked\r\n", "2\r\n{"),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "2\r\n{}",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "2\r\n{}\r",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "0\r\nTrailer: value\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "0\r\n\r",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: gzip\r\n",
                "2\r\n{}\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: gzip\r\nTransfer-Encoding: chunked\r\n",
                "2\r\n{}\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked, gzip\r\n",
                "2\r\n{}\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: xchunked\r\n",
                "2\r\n{}\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding : chunked\r\n",
                "2\r\n{}\r\n0\r\n\r\n",
            ),
            ("POST /tokenize", "X-Folded\r\nContent-Length: 2\r\n", "{}"),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "+2\r\n{}\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                " 2\r\n{}\r\n0\r\n\r\n",
            ),
            ("GET /health", &smuggled, next),
            ("GET /v1/health", &smuggled, next),
            ("OPTIONS /tokenize", &smuggled, next),
            ("GET /health", "Content-Length: +0\r\n", ""),
            ("GET /health", "Content-Length\u{00a0}: 24\r\n", next),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "f \r\n{\"content\":\"x\"}\r\n0\r\n\r\n",
            ),
            (
                "POST /tokenize",
                "Transfer-Encoding: chunked\r\n",
                "f\t\r\n{\"content\":\"x\"}\r\n0\r\n\r\n",
            ),
        ]
        .into_iter()
        .map(|(start, headers, body)| (start, headers.to_owned(), body.to_owned()))
        .chain(whitespace_frames)
        {
            let mut stream = TcpStream::connect(addr).expect("connect");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            stream
                .write_all(format!("{start} HTTP/1.1\r\n{headers}\r\n{body}").as_bytes())
                .unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut reader = BufReader::new(stream);
            let (status, head, _) = response(&mut reader);
            if status != 400 || !head.contains("Connection: close") {
                accepted.push(format!("{start} {headers:?} {body:?}: {head}"));
            }
            let mut byte = [0];
            assert_eq!(
                reader
                    .read(&mut byte)
                    .expect("the refused connection closes"),
                0,
                "no next request after malformed framing: {start} {headers:?}"
            );
        }
        assert!(
            accepted.is_empty(),
            "malformed framing accepted: {accepted:?}"
        );
        for (head, status) in [
            (
                "POST /tokenize HTTP/1.1\r\nContent-Length: 2, 3\r\n\r\n",
                400,
            ),
            ("GET /health HTTP/1.1\r\nContent-Length: 1024\r\n\r\n", 200),
        ] {
            assert_eq!(answer_before_the_body(addr, head), status);
        }

        for (headers, body) in [
            (
                format!("Content-Length: {0}\r\ncontent-length: {0}\r\n", body.len()),
                body.to_owned(),
            ),
            (
                format!("Content-Length: {0}, {0}\r\n", body.len()),
                body.to_owned(),
            ),
            (
                format!("Content-Length: \t{0} \t,\t {0}\t \r\n", body.len()),
                body.to_owned(),
            ),
            (
                "Transfer-Encoding: chunked\r\n".to_owned(),
                format!(
                    "{:x};x=y\r\n{body}\r\n0\r\nTrailer: value\r\n\r\n",
                    body.len()
                ),
            ),
            (
                "Transfer-Encoding: chunked\r\n".to_owned(),
                format!(
                    "{:x} \t;x=y\r\n{body}\r\n0\r\nTrailer: value\r\n\r\n",
                    body.len()
                ),
            ),
        ] {
            let mut stream = TcpStream::connect(addr).expect("connect");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            stream.write_all(format!("POST /tokenize HTTP/1.1\r\n{headers}\r\n{body}GET /health HTTP/1.1\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
            let mut reader = BufReader::new(stream);
            let (status, head, _) = response(&mut reader);
            assert_eq!(status, 200, "valid framing: {headers:?}");
            assert!(head.contains("Connection: keep-alive"));
            assert_eq!(response(&mut reader).0, 200, "the next request survives");
        }

        for (path, body) in [
            ("/tokenize", r#"{"content":"x"}"#),
            (
                "/completion",
                r#"{"prompt":"x","n_predict":1,"stream":true}"#,
            ),
        ] {
            for length in [
                "Content-Length: 1\r\n",
                "Content-Length: 1\r\nContent-Length: 2\r\n",
                "content-length: 1, 2\r\n",
                "cOnTeNt-LeNgTh: invalid\r\n",
                "Content-Length: \r\n",
            ] {
                let mut stream = TcpStream::connect(addr).expect("connect");
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream.write_all(format!("POST {path} HTTP/1.1\r\n{length}Transfer-Encoding: ChUnKeD\r\nExpect: 100-continue\r\n\r\n{:x}\r\n{body}\r\n0\r\n\r\n{next}", body.len()).as_bytes()).unwrap();
                let mut reader = BufReader::new(stream);
                assert_eq!(response(&mut reader).0, 100, "chunked takes precedence");
                let mut raw = String::new();
                reader
                    .read_to_string(&mut raw)
                    .expect("the response ends with EOF");
                let (head, _) = raw.split_once("\r\n\r\n").expect("response head");
                assert!(
                    head.starts_with("HTTP/1.1 200 "),
                    "{path} {length:?}: {head}"
                );
                assert!(
                    head.contains("Connection: close"),
                    "{path} {length:?}: {head}"
                );
                assert_eq!(
                    raw.matches("HTTP/1.1 ").count(),
                    1,
                    "no pipelined response: {raw}"
                );
                if path == "/completion" {
                    assert!(raw.ends_with("0\r\n\r\n"), "the stream finishes: {raw}");
                }
            }
        }
    }

    /// A body sent gzip or deflate is answered as the bytes it decodes to. A body that does not
    /// decode, carries bytes past its stream or decodes past the body limit is a 400; another
    /// coding, or two, is a 415 whose `Accept-Encoding` names the ones the server reads, sent
    /// before the body is asked for. A request with no body reads as sent.
    #[test]
    fn compressed_request_bodies_are_read_decoded() {
        use crate::http::{MAX_BODY, read_test_response as response};
        use flate2::Compression;
        use flate2::write::{GzEncoder, ZlibEncoder};
        use std::io::{BufReader, Write};
        use std::net::TcpStream;

        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let send = |head: &str, body: &[u8]| {
            let mut stream = TcpStream::connect(addr).expect("connect");
            stream
                .set_read_timeout(Some(Duration::from_secs(60)))
                .unwrap();
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
            let (status, head, body) = response(&mut BufReader::new(stream));
            (
                status,
                head,
                String::from_utf8(body).expect("a UTF-8 answer"),
            )
        };
        let post = |coding: &str, body: &[u8]| {
            let head = format!(
                "POST /tokenize HTTP/1.1\r\nConnection: close\r\nContent-Encoding: {coding}\r\n\
                 Content-Length: {}\r\n\r\n",
                body.len()
            );
            send(&head, body)
        };
        let gzip = |b: &[u8]| {
            let mut e = GzEncoder::new(Vec::new(), Compression::fast());
            e.write_all(b).unwrap();
            e.finish().unwrap()
        };
        let deflate = |b: &[u8]| {
            let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());
            e.write_all(b).unwrap();
            e.finish().unwrap()
        };

        let json = br#"{"content":"hello"}"#;
        let (status, _, plain) = post("identity", json);
        assert_eq!(status, 200, "{plain}");
        for (coding, body) in [
            ("gzip", gzip(json)),
            ("X-GZIP", gzip(json)),
            ("deflate", deflate(json)),
            ("gzip,, identity", gzip(json)),
            (
                "gzip",
                [gzip(br#"{"content":"#), gzip(br#""hello"}"#)].concat(),
            ),
        ] {
            let (status, _, answer) = post(coding, &body);
            assert_eq!((status, answer.as_str()), (200, plain.as_str()), "{coding}");
        }
        let body = gzip(json);
        let chunked = [
            format!("{:x}\r\n", body.len()).as_bytes(),
            &body,
            b"\r\n0\r\n\r\n",
        ]
        .concat();
        let (status, _, answer) = send(
            "POST /tokenize HTTP/1.1\r\nConnection: close\r\nContent-Encoding: gzip\r\n\
             Transfer-Encoding: chunked\r\n\r\n",
            &chunked,
        );
        assert_eq!((status, answer.as_str()), (200, plain.as_str()), "chunked");

        let stream = deflate(json);
        let mebibyte = gzip(&vec![0; 1 << 20]);
        let mut past_deflate = deflate(&vec![0; MAX_BODY + (1 << 20)]);
        *past_deflate.last_mut().expect("a checksum") ^= 1;
        let past = |name: &str| {
            format!(
                "body too large: the {name} body decodes past the limit of {MAX_BODY} bytes ({} MiB)",
                MAX_BODY >> 20
            )
        };
        for (coding, body, says) in [
            (
                "gzip",
                json.to_vec(),
                "the body does not decode as gzip: ".to_owned(),
            ),
            (
                "deflate",
                stream[..stream.len() - 1].to_vec(),
                "the body does not decode as deflate: ".to_owned(),
            ),
            (
                "deflate",
                [&stream[..], b"{}"].concat(),
                "2 bytes follow the body's deflate stream".to_owned(),
            ),
            // A body that decodes to the limit itself is read: the JSON reader refuses its zeros.
            (
                "gzip",
                mebibyte.repeat(MAX_BODY >> 20),
                "invalid JSON body: ".to_owned(),
            ),
            // Decoding stops past the limit, before the bytes that do not decode (a gzip member
            // that is not one, a deflate checksum that is wrong).
            (
                "gzip",
                [mebibyte.repeat((MAX_BODY >> 20) + 1), b"invalid".to_vec()].concat(),
                past("gzip"),
            ),
            ("deflate", past_deflate, past("deflate")),
        ] {
            let (status, _, answer) = post(coding, &body);
            let v: Value = serde_json::from_str(&answer).expect("an error object");
            let message = v["error"]["message"].as_str().expect("a message");
            assert_eq!(status, 400, "{coding}: {answer}");
            assert!(message.starts_with(&says), "{coding}: {message}");
        }

        for coding in ["br", "deflate, gzip"] {
            let (status, head, answer) = send(
                &format!(
                    "POST /tokenize HTTP/1.1\r\nContent-Encoding: {coding}\r\n\
                     Expect: 100-continue\r\nContent-Length: 20\r\n\r\n"
                ),
                b"",
            );
            assert_eq!(status, 415, "refused before 100 Continue: {head}");
            assert!(
                head.contains("\r\nAccept-Encoding: gzip, deflate\r\n"),
                "{head}"
            );
            let v: Value = serde_json::from_str(&answer).expect("an error object");
            assert_eq!(
                v["error"]["message"],
                format!(
                    "unsupported content-encoding {coding:?}: the server reads a body in one coding, \
                     gzip, deflate or identity"
                )
            );
        }

        for length in ["", "Content-Length: 0\r\n"] {
            let (status, _, answer) = send(
                &format!(
                    "GET /v1/models HTTP/1.1\r\nConnection: close\r\nContent-Encoding: br\r\n\
                     {length}\r\n"
                ),
                b"",
            );
            assert_eq!(status, 200, "{length:?}: {answer}");
        }
    }

    /// `POST /shutdown` needs the key even from a loopback peer: without one
    /// it is the 401 and the server keeps serving; with one it is the 200
    /// and the stop, as without keys.
    #[test]
    fn shutdown_needs_the_key_even_from_loopback() {
        let mut keys = ApiKeys::default();
        keys.add("--api-key", KEY).unwrap();
        let (addr, state, ended) = super::testserve::spawn_keyed(keys);

        let (status, body) = super::testserve::roundtrip(addr, "POST", "/shutdown", &[], "");
        assert_eq!(status, 401, "no key: {body}");
        assert!(
            ended.recv_timeout(Duration::from_millis(200)).is_err(),
            "the 401 stops nothing"
        );

        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/shutdown", &auth(BEARER), "");
        assert_eq!(status, 200, "the keyed post: {body}");
        match ended.recv_timeout(Duration::from_secs(5)) {
            Ok(End::Shutdown(Stop::Posted(peer))) => {
                assert!(peer.ip().is_loopback(), "the stop names its peer: {peer}")
            }
            Ok(_) => panic!("the end channel carried another end"),
            Err(e) => panic!("the end channel carries no shutdown: {e}"),
        }
        assert!(
            state.shared.wait_loop_end(Duration::from_secs(5)),
            "the engine thread leaves its loop once the stop reaches it"
        );
    }

    /// The Anthropic Messages route takes the key the way Claude Code sends
    /// it: `x-api-key` on both `/v1/messages` and its `count_tokens`.
    #[test]
    fn the_anthropic_messages_route_takes_the_x_api_key_header() {
        let mut keys = ApiKeys::default();
        keys.add("--api-key", KEY).unwrap();
        let (addr, _state, _ended) = super::testserve::spawn_keyed(keys);
        let messages = r#"{"max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#;
        let counting = r#"{"messages":[{"role":"user","content":"hi"}]}"#;

        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/v1/messages", &[], messages);
        assert_eq!(status, 401, "no key: {body}");
        for (path, body_of) in [
            ("/v1/messages", messages),
            ("/v1/messages/count_tokens", counting),
        ] {
            let (status, text) =
                super::testserve::roundtrip(addr, "POST", path, &xkey(KEY), body_of);
            assert_eq!(status, 200, "{path} with x-api-key: {text}");
        }
    }

    /// A key never reaches an answer: with a known key set, a 401 (its head
    /// and body), `/props`, `/slots`, `/metrics`, `/health` and a 400 error
    /// body carry none of the key's bytes, and the one stderr line a
    /// refusal writes (composed by [`deny_line`], pinned below) names the
    /// method, path and peer only.
    #[test]
    fn a_key_never_reaches_an_answer() {
        let mut keys = ApiKeys::default();
        keys.add("--api-key", KEY).unwrap();
        let (addr, _state, _ended) = super::testserve::spawn_keyed(keys);

        let ask = r#"{"prompt":"I believe the meaning of life is","max_tokens":4}"#;
        let (status, head, body) =
            super::testserve::roundtrip_head(addr, "POST", "/completion", &[], ask);
        assert_eq!(status, 401, "{body}");
        for text in [&head, &body] {
            assert!(!text.contains(KEY), "the 401 carries no key: {text}");
        }
        for path in ["/props", "/slots", "/metrics", "/health"] {
            let (status, _head, body) =
                super::testserve::roundtrip_head(addr, "GET", path, &auth(BEARER), "");
            assert_eq!(status, 200, "{path}: {body}");
            assert!(!body.contains(KEY), "{path} carries no key: {body}");
        }
        let (status, _head, body) = super::testserve::roundtrip_head(
            addr,
            "POST",
            "/tokenize",
            &auth(BEARER),
            "{ not json",
        );
        assert_eq!(status, 400, "an error body to read: {body}");
        assert!(!body.contains(KEY), "the error body carries no key: {body}");

        let line = deny_line("POST", "/completion", "127.0.0.1:54094", Denied::Key);
        assert!(
            line.contains("POST /completion") && line.contains("127.0.0.1:54094"),
            "{line}"
        );
        assert!(
            !line.contains(KEY),
            "the refusal's line carries no key: {line}"
        );
    }

    /// `/props`' `engine.args` holds back the key: the token after
    /// `--api-key` is `<redacted>`, `--api-key-file`'s token (a file name,
    /// not a key) and every other token stay verbatim, and an argv with no
    /// key flag is unchanged.
    #[test]
    fn argv_redacted_holds_the_key_back() {
        let argv = [
            "bloomery-serve",
            "--host",
            "0.0.0.0",
            "--api-key",
            KEY,
            "--api-key-file",
            "/etc/bloomery/keys",
            "--port",
            "8080",
        ];
        assert_eq!(
            argv_redacted(argv.into_iter().map(str::to_owned)),
            [
                "bloomery-serve",
                "--host",
                "0.0.0.0",
                "--api-key",
                "<redacted>",
                "--api-key-file",
                "/etc/bloomery/keys",
                "--port",
                "8080",
            ]
        );
        let plain = ["bloomery-serve", "--port", "8080"];
        let redacted = argv_redacted(plain.into_iter().map(str::to_owned));
        assert_eq!(redacted, plain, "no key flag, nothing held back");
        assert!(!redacted.join(" ").contains(KEY));
    }

    /// With no key set, every route answers exactly as it does with the
    /// check passed: a server with no keys and a keyed server asked with the
    /// valid key answer the same fixed set — every route the server has —
    /// with the same status, content type and body. The bodies are compared
    /// as JSON with the two leaves two server instances cannot share pinned
    /// to a constant (`created`, the bind time) or dropped (`timings`, the
    /// measured walls); `/metrics` is scraped before any generation for the
    /// same reason (its counters would hold measured seconds). The stream
    /// and `OPTIONS` bodies are text, compared whole.
    #[test]
    fn with_no_keys_set_every_route_answers_as_it_does_with_the_check_passed() {
        let plain = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let mut keys = ApiKeys::default();
        keys.add("--api-key", KEY).unwrap();
        let keyed = super::testserve::spawn_keyed(keys);
        let bearer = auth(BEARER);

        let chat = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
        let completion = r#"{"prompt":"ab","max_tokens":2}"#;
        let stream = r#"{"prompt":"ab","max_tokens":2,"stream":true}"#;
        let tokenize = r#"{"content":"hello"}"#;
        // `/metrics` first: its counters hold measured seconds once a
        // generation has run.
        let set: &[(&str, &str, &str)] = &[
            ("GET", "/metrics", ""),
            ("GET", "/health", ""),
            ("GET", "/v1/models", ""),
            ("GET", "/props", ""),
            ("GET", "/slots", ""),
            ("POST", "/tokenize", tokenize),
            ("POST", "/completion", completion),
            ("POST", "/v1/chat/completions", chat),
            ("POST", "/completion", stream),
            ("OPTIONS", "/v1/chat/completions", ""),
        ];
        for (method, path, body) in set {
            let (s0, h0, b0) = super::testserve::roundtrip_head(plain.0, method, path, &[], body);
            let (s1, h1, b1) =
                super::testserve::roundtrip_head(keyed.0, method, path, &bearer, body);
            assert_eq!(
                s0, s1,
                "{method} {path}: the check must not change the status"
            );
            let ctype = |h: &str| {
                h.lines()
                    .find(|l| l.starts_with("Content-Type:"))
                    .unwrap_or("no content type")
                    .to_owned()
            };
            assert_eq!(ctype(&h0), ctype(&h1), "{method} {path}: the content type");
            let norm = |b: &str| normalize(b);
            assert_eq!(norm(&b0), norm(&b1), "{method} {path}: the body");
            assert_ne!(s0, 401, "{method} {path}: no key set, no 401");
        }

        /// One body as the comparison takes it: JSON with `created`, `id`
        /// and `seed` (the bind time, the wall-clock-mixed request id, and
        /// the default seed drawn from both) pinned and `timings` dropped; a
        /// stream the same, chunk by chunk;
        /// anything else (`OPTIONS`' empty body) whole.
        fn normalize(body: &str) -> String {
            if let Ok(v) = serde_json::from_str::<Value>(body) {
                return pinned(v).to_string();
            }
            body.lines()
                .map(|l| {
                    let Some(payload) = l.strip_prefix("data: ") else {
                        return l.to_owned();
                    };
                    let v = serde_json::from_str::<Value>(payload).expect("a stream chunk is JSON");
                    format!("data: {v}", v = pinned(v))
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        /// [`normalize`]'s one JSON value: the clock-bound leaves pinned, the
        /// measured walls dropped.
        fn pinned(mut v: Value) -> Value {
            fn walk(v: &mut Value) {
                // An array's objects carry their own clock leaves (/v1/models' `data[].created`).
                if let Value::Array(a) = v {
                    a.iter_mut().for_each(walk);
                    return;
                }
                let Value::Object(o) = v else { return };
                o.remove("timings");
                for leaf in ["created", "id", "seed"] {
                    if let Some(c) = o.get_mut(leaf) {
                        *c = Value::from(0);
                    }
                }
                for (_k, x) in o.iter_mut() {
                    walk(x);
                }
            }
            walk(&mut v);
            v
        }
    }

    /// The media mock whose two slots take it in turns, parked as `park`.
    struct Turned(crate::MockEngine, Park);

    impl Engine for Turned {
        fn tokenizer(&self) -> Arc<dyn crate::Tokenizer> {
            self.0.tokenizer()
        }
        fn media_model(&self) -> Option<crate::media::SharedMediaModel> {
            self.0.media_model()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), crate::EngineError> {
            self.0.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, crate::EngineError> {
            self.0.next(last, out)
        }
        fn slots(&self) -> usize {
            2
        }
        fn turns(&self) -> Option<Park> {
            Some(self.1)
        }
        fn reset(&mut self) -> Result<(), crate::EngineError> {
            self.0.reset()
        }
        fn ctx_max(&self) -> usize {
            self.0.ctx_max()
        }
        fn describe(&self) -> String {
            self.0.describe()
        }
    }

    /// Slots that park a request as its ids cannot feed its image again, so
    /// an engine of images whose slots take it in turns so is refused by name
    /// at two slots; one slot parks nothing, and slots that park states, or an
    /// engine of no images, serve.
    #[test]
    fn images_refuse_slots_parked_as_ids() {
        let media = || crate::MockEngine::new(64).with_media().0;
        let slots = |n: usize| SlotConfig {
            parallel: n,
            ..SlotConfig::default()
        };
        match check_slots(&Turned(media(), Park::Ids), &slots(2)) {
            Err(ServeError::Slots(why)) => assert!(why.contains("takes images"), "{why}"),
            other => panic!("{:?}", other.map_err(|e| e.to_string())),
        }
        assert!(check_slots(&Turned(media(), Park::Ids), &slots(1)).is_ok());
        let states = Park::States { budget: 1 << 20 };
        assert!(check_slots(&Turned(media(), states), &slots(2)).is_ok());
        let text = Turned(crate::MockEngine::new(64), Park::Ids);
        assert!(check_slots(&text, &slots(2)).is_ok());
    }

    /// A request that answers once the generation is whole sends nothing while
    /// it runs, so a client that left is found by the probe its sink runs at
    /// each event ([`probe_client`], [`run_whole`]): over real sockets, and
    /// with an engine the test paces a step at a time.
    mod gone_client {
        use std::io::{self, Read, Write};
        use std::net::{SocketAddr, TcpListener, TcpStream};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        use crate::api::{probe_client, relock, testserve};
        use crate::sched::Use;
        use crate::{Engine, EngineError, MockEngine, SavedState, StateError, Tokenizer};

        /// The mock engine with every `next` and every snapshot held at a
        /// gate: the call is counted, then waits for a permit the test sends,
        /// so the test decides how many steps the generation takes and where
        /// a move between slots that take it in turns stands.
        struct Gated {
            inner: MockEngine,
            calls: Arc<AtomicUsize>,
            permits: mpsc::Receiver<()>,
            prefilled: Arc<AtomicUsize>,
        }

        impl Gated {
            fn gate(&self) -> Result<(), EngineError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.permits
                    .recv()
                    .map_err(|_| EngineError("the test's gate closed".to_owned()))
            }
        }

        impl Engine for Gated {
            fn tokenizer(&self) -> Arc<dyn Tokenizer> {
                self.inner.tokenizer()
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
                self.prefilled.fetch_add(1, Ordering::SeqCst);
                self.inner.prefill(ids)
            }
            fn next(
                &mut self,
                last: u32,
                logits_out: Option<&mut [f32]>,
            ) -> Result<u32, EngineError> {
                self.gate()?;
                self.inner.next(last, logits_out)
            }
            fn reset(&mut self) -> Result<(), EngineError> {
                self.inner.reset()
            }
            fn ctx_max(&self) -> usize {
                self.inner.ctx_max()
            }
            fn describe(&self) -> String {
                "the gated mock".to_owned()
            }
            fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
                self.gate()?;
                self.inner.save_state(out)
            }
            fn restore_state(&mut self, input: &mut dyn Read) -> Result<SavedState, StateError> {
                self.inner.restore_state(input)
            }
        }

        #[cfg(unix)]
        fn reset(client: TcpStream) {
            use std::os::fd::AsRawFd;
            let linger = libc::linger {
                l_onoff: 1,
                l_linger: 0,
            };
            // SAFETY: the socket is live and linger points to a value of the stated size.
            let rc = unsafe {
                libc::setsockopt(
                    client.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_LINGER,
                    (&linger as *const libc::linger).cast(),
                    std::mem::size_of_val(&linger) as libc::socklen_t,
                )
            };
            assert_eq!(rc, 0, "linger: {}", io::Error::last_os_error());
            drop(client);
        }

        /// A greedy `/completion` of `prompt` for `n_predict` tokens, its answer unread.
        fn complete(addr: SocketAddr, prompt: &str, n_predict: usize) -> TcpStream {
            let body =
                format!(r#"{{"prompt":"{prompt}","temperature":0,"n_predict":{n_predict}}}"#);
            testserve::send_request(addr, "POST", "/completion", &[], &body)
        }

        /// The whole answer `s` reads, a 200: its JSON body.
        fn answer(mut s: TcpStream) -> serde_json::Value {
            s.set_read_timeout(Some(Duration::from_secs(3)))
                .expect("timeout");
            let mut raw = String::new();
            s.read_to_string(&mut raw).expect("answer");
            assert!(raw.starts_with("HTTP/1.0 200"), "{raw}");
            serde_json::from_str(raw.split_once("\r\n\r\n").expect("head").1).expect("JSON")
        }

        /// An opened SSE stream holds its key's one place until completion
        /// or disconnect, including while the next engine step is blocked.
        #[cfg(unix)]
        #[test]
        fn a_stream_keeps_its_key_place_until_completion_or_disconnect() {
            use std::io::{BufRead, BufReader};

            for disconnect in [false, true] {
                let calls = Arc::new(AtomicUsize::new(0));
                let (permit, permits) = mpsc::channel();
                let (keys, _) = crate::flag::key_file("stream", "sk-stream inflight=1\n");
                let mut config = testserve::mock_config();
                config.api_keys = keys.expect("the key file");
                let (addr, _state, _ended) = testserve::spawn(
                    Box::new(Gated {
                        inner: MockEngine::new(1024),
                        calls: Arc::clone(&calls),
                        permits,
                        prefilled: Arc::new(AtomicUsize::new(0)),
                    }),
                    config,
                );
                let auth = [("Authorization", "Bearer sk-stream")];
                let client = testserve::send_request(
                    addr,
                    "POST",
                    "/completion",
                    &auth,
                    r#"{"prompt":"abab","temperature":0,"n_predict":2,"stream":true}"#,
                );
                client
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .expect("timeout");
                wait_for("the stream's first step", || {
                    calls.load(Ordering::SeqCst) == 1
                });
                permit.send(()).expect("release the first token");
                wait_for("the stream's next step", || {
                    calls.load(Ordering::SeqCst) == 2
                });
                let mut read = BufReader::new(client);
                loop {
                    let mut line = String::new();
                    assert_ne!(read.read_line(&mut line).expect("stream head or event"), 0);
                    if line.starts_with("data: ") {
                        break;
                    }
                }
                let probe = || {
                    testserve::roundtrip_head(
                        addr,
                        "POST",
                        "/tokenize",
                        &auth,
                        r#"{"content":"ab"}"#,
                    )
                };
                let (status, head, body) = probe();
                assert_eq!(
                    status, 429,
                    "the opened stream holds its place: {head}{body}"
                );
                assert!(head.contains("Retry-After: 1\r\n"), "{head}");
                if disconnect {
                    reset(read.into_inner());
                } else {
                    permit.send(()).expect("finish the stream");
                    let mut rest = String::new();
                    read.read_to_string(&mut rest).expect("the stream ends");
                    assert!(rest.contains(r#""stop":true"#), "{rest}");
                    assert_eq!(probe().0, 200, "completion returns the place before EOF");
                    continue;
                }
                permit
                    .send(())
                    .expect("let the engine observe the disconnect");
                wait_for("the disconnected stream's place", || probe().0 == 200);
            }
        }

        /// A request its client leaves while the engine is in its prefill
        /// holds its key's one place until the engine leaves that prefill:
        /// a cancelled background response and a whole request whose
        /// connection resets both keep it while the prefill is held, though
        /// their request side ends at its next disconnect check, and give it back
        /// once the prefill ends.
        #[cfg(unix)]
        #[test]
        fn a_left_request_keeps_its_key_place_until_the_engine_leaves_it() {
            use crate::mock::{DraftMock, Hold};

            for background in [true, false] {
                let (keys, _) = crate::flag::key_file("left", "sk-left inflight=1\n");
                let mut config = testserve::mock_config();
                config.api_keys = keys.expect("the key file");
                let hold = Arc::new(Hold::default());
                let (addr, _state, _ended) = testserve::spawn(
                    Box::new(DraftMock::new(64).holding(Arc::clone(&hold))),
                    config,
                );
                let auth = [("Authorization", "Bearer sk-left")];
                let probe = || {
                    testserve::roundtrip(addr, "POST", "/tokenize", &auth, r#"{"content":"ab"}"#).0
                };
                if background {
                    let (status, body) = testserve::roundtrip(
                        addr,
                        "POST",
                        "/v1/responses",
                        &auth,
                        r#"{"input":"abab","background":true,"max_output_tokens":2}"#,
                    );
                    assert_eq!(status, 200, "{body}");
                    let v: serde_json::Value = serde_json::from_str(&body).expect("JSON");
                    let id = v["id"].as_str().expect("the response's id");
                    assert!(hold.reached(1, Duration::from_secs(10)));
                    let cancel = format!("/v1/responses/{id}/cancel");
                    let (status, body) = testserve::roundtrip(addr, "POST", &cancel, &auth, "");
                    assert_eq!(status, 200, "{body}");
                } else {
                    let client = testserve::send_request(
                        addr,
                        "POST",
                        "/completion",
                        &auth,
                        r#"{"prompt":"abab","n_predict":2}"#,
                    );
                    assert!(hold.reached(1, Duration::from_secs(10)));
                    reset(client);
                }
                let until = Instant::now() + Duration::from_secs(1);
                while Instant::now() < until {
                    assert_eq!(
                        probe(),
                        429,
                        "the held prefill keeps the place (background={background})"
                    );
                    std::thread::sleep(Duration::from_millis(50));
                }
                hold.open();
                wait_for("the place after the prefill", || probe() == 200);
            }
        }

        #[test]
        fn lifecycle_half_closed_request_receives_its_whole_answer() {
            let calls = Arc::new(AtomicUsize::new(0));
            let (permit, permits) = mpsc::channel();
            let (addr, state, _ended) = testserve::spawn(
                Box::new(Gated {
                    inner: MockEngine::new(1024),
                    calls: Arc::clone(&calls),
                    permits,
                    prefilled: Arc::new(AtomicUsize::new(0)),
                }),
                testserve::mock_config(),
            );
            let client = complete(addr, "abab", 1);
            wait_for("the first step", || calls.load(Ordering::SeqCst) == 1);
            client
                .shutdown(std::net::Shutdown::Write)
                .expect("half close");
            permit.send(()).expect("release");
            let body = answer(client);
            assert_eq!(body["content"], "a");
            assert_eq!(body["tokens_predicted"], 1);
            assert_eq!(
                relock(&state.shared.stats).n_cancelled_before_prompt_total,
                0
            );
        }

        #[cfg(unix)]
        #[test]
        fn lifecycle_reset_while_queued_never_prefills() {
            let calls = Arc::new(AtomicUsize::new(0));
            let prefilled = Arc::new(AtomicUsize::new(0));
            let (permit, permits) = mpsc::channel();
            let server = super::super::Server::bind_with(
                "127.0.0.1:0",
                Box::new(Gated {
                    inner: MockEngine::new(1024),
                    calls: Arc::clone(&calls),
                    permits,
                    prefilled: Arc::clone(&prefilled),
                }),
                testserve::mock_config(),
                crate::SlotConfig {
                    queue_depth: Some(1),
                    ..crate::SlotConfig::default()
                },
            )
            .expect("bind");
            let addr = server.local_addr().expect("addr");
            let (state, _ended) = server.start().expect("start");
            let first = complete(addr, "abab", 1);
            wait_for("the first request", || calls.load(Ordering::SeqCst) == 1);
            let second = complete(addr, "cdcd", 1);
            wait_for("the queued request", || {
                relock(&state.shared.board).waiting() == 1
            });
            reset(second);
            wait_for("the queued connection's release", || {
                state.shared.answers.in_flight() == 1
            });
            assert_eq!(
                relock(&state.shared.board).waiting(),
                0,
                "the reset connection still occupies the queue"
            );
            let third = complete(addr, "efef", 1);
            wait_for(
                "the next live request queued before the engine returns",
                || relock(&state.shared.board).waiting() == 1,
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            permit.send(()).expect("release first");
            answer(first);
            wait_for("the next live request", || {
                calls.load(Ordering::SeqCst) == 2
            });
            permit.send(()).expect("release next");
            answer(third);
            assert_eq!(
                prefilled.load(Ordering::SeqCst),
                2,
                "a reset request reached prefill"
            );
            let (status, metrics) = testserve::roundtrip(addr, "GET", "/metrics", &[], "");
            assert_eq!(status, 200, "{metrics}");
            assert!(
                metrics.contains("llamacpp:requests_cancelled_before_prompt_total 1\n"),
                "{metrics}"
            );
        }

        #[cfg(unix)]
        #[test]
        fn lifecycle_concurrent_reset_never_starts_a_prompt() {
            use crate::api::Client;
            use crate::genloop::GenParams;
            use crate::media::Held;
            use crate::worker::{ReqName, Submit};
            use std::sync::Mutex;

            let prefilled = Arc::new(AtomicUsize::new(0));
            let (permit, permits) = mpsc::channel();
            let (addr, state, _ended) = testserve::spawn(
                Box::new(Gated {
                    inner: MockEngine::new(1024),
                    calls: Arc::new(AtomicUsize::new(0)),
                    permits,
                    prefilled: Arc::clone(&prefilled),
                }),
                testserve::mock_config(),
            );
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let peer = TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
            let (stream, _) = listener.accept().expect("accept");
            let client = Client::of(stream, &super::super::closing());
            let (probed, consumed) = mpsc::channel();
            let (resume, resumed) = mpsc::channel();
            reset(peer);
            let observer = Arc::clone(&client);
            let pause = Mutex::new(Some((probed, resumed)));
            let (checked, checking) = mpsc::channel();
            let live = Arc::new(AtomicUsize::new(0));
            super::super::admit(
                &Arc::new(move |_: TcpStream| {
                    testserve::CLIENT_PROBE.with_borrow_mut(|hook| *hook = relock(&pause).take());
                    wait_for("the reset error", || observer.check().is_err());
                    checked.send(()).expect("the checker finished");
                }),
                &live,
                &Arc::new(AtomicUsize::new(0)),
                0,
                client
                    .stream
                    .as_ref()
                    .and_then(|s| s.try_clone().ok())
                    .expect("the checker's connection"),
            );
            consumed
                .recv_timeout(Duration::from_secs(3))
                .expect("the first observer consumed the reset");
            let (events, received) = mpsc::channel();
            relock(&state.shared.board)
                .enqueue(
                    Held::from(vec![97, 98, 97, 98]),
                    Submit {
                        p: GenParams {
                            n_predict: 1,
                            ..super::super::default_params(&state)
                        },
                        prompt: serde_json::Value::Null,
                        settings: serde_json::Value::Null,
                        media: Vec::new(),
                        sampler: None,
                        events: crate::worker::Events::new(events, 0, None),
                        name: ReqName {
                            id: 0,
                            route: "/completion".to_owned(),
                            choice: 0,
                            choices: 1,
                        },
                        client,
                    },
                )
                .unwrap_or_else(|_| panic!("queue refused"));
            state.shared.work.notify_one();
            wait_for("the worker's admission", || {
                let board = relock(&state.shared.board);
                board.waiting() == 0 && board.free() == 0
            });
            let admitted = received.recv_timeout(Duration::from_millis(200));
            permit.send(()).expect("release any incorrect prefill");
            resume.send(()).expect("publish the reset");
            checking
                .recv_timeout(Duration::from_secs(3))
                .expect("the first checker");
            wait_for("the checker's connection release", || {
                live.load(Ordering::SeqCst) == 0
            });
            assert!(
                matches!(admitted, Err(mpsc::RecvTimeoutError::Timeout)),
                "the worker admitted a request while its reset was being observed"
            );
            assert!(
                matches!(
                    received.recv_timeout(Duration::from_secs(3)),
                    Err(mpsc::RecvTimeoutError::Disconnected)
                ),
                "the reset request sent an event"
            );
            wait_for("the cancelled slot's release", || {
                relock(&state.shared.board).free() == 1
            });
            assert_eq!(prefilled.load(Ordering::SeqCst), 0);
            let (status, metrics) = testserve::roundtrip(addr, "GET", "/metrics", &[], "");
            assert_eq!(status, 200, "{metrics}");
            assert!(
                metrics.contains("llamacpp:requests_cancelled_before_prompt_total 1\n"),
                "{metrics}"
            );
        }

        /// Two slots that take the engine in turns, A running on slot 0 and
        /// B's ended state parked idle on slot 1: a request for slot 1 whose
        /// client left while it waited is cancelled before the engine moves,
        /// and one whose client resets while A's snapshot is taken for the
        /// move takes B's handed-back state with its prompt. A completes and
        /// the server serves on.
        #[cfg(unix)]
        #[test]
        fn lifecycle_resets_around_a_turn_switch_leave_the_turns_serving() {
            use crate::api::Server;
            use crate::swap::Turn;
            use crate::{Park, SlotConfig, SwapEngine};

            let calls = Arc::new(AtomicUsize::new(0));
            let prefilled = Arc::new(AtomicUsize::new(0));
            let (permit, permits) = mpsc::channel();
            let gated = Gated {
                inner: MockEngine::new(1024),
                calls: Arc::clone(&calls),
                permits,
                prefilled: Arc::clone(&prefilled),
            };
            let swap = SwapEngine::new(Box::new(gated), 2, Park::States { budget: 1 << 20 })
                .unwrap_or_else(|e| panic!("swap engine: {e}"));
            let slots = SlotConfig {
                parallel: 2,
                ..SlotConfig::default()
            };
            let server = Server::bind_with(
                "127.0.0.1:0",
                Box::new(swap),
                testserve::mock_config(),
                slots,
            )
            .expect("bind");
            let addr = server.local_addr().expect("addr");
            let (state, _ended) = server.start().expect("start");
            // Lets the `k`-th gated call run once it waits.
            let release = |k: usize, what: &str| {
                wait_for(what, || calls.load(Ordering::SeqCst) == k);
                permit.send(()).expect(what);
            };
            let queued = |what: &str| wait_for(what, || relock(&state.shared.board).waiting() == 1);
            let a = complete(addr, "abab", 40);
            wait_for("A's prompt step", || calls.load(Ordering::SeqCst) == 1);
            let b = complete(addr, "cdcd", 1);
            queued("B queued");
            release(1, "A's prompt step");
            release(2, "A's second step");
            release(3, "A parked for B");
            release(4, "B's prompt step");
            assert_eq!(answer(b)["content"], "c");
            release(5, "B parked idle for A");
            wait_for("A's third step", || calls.load(Ordering::SeqCst) == 6);
            assert_eq!(
                relock(&state.shared.turns).as_deref(),
                Some(&[Turn::Running, Turn::Idle][..]),
                "the fixture: A runs on slot 0 and slot 1 is idle"
            );
            assert_eq!(
                relock(&state.shared.stats).parked_bytes,
                16 + 4 * 4,
                "the fixture: B's four positions parked"
            );

            let gone = complete(addr, "efef", 1);
            queued("the request that leaves queued");
            reset(gone);
            wait_for("its connection's release", || {
                state.shared.answers.in_flight() == 1
            });
            release(6, "A's third step");
            wait_for("A's fourth step", || calls.load(Ordering::SeqCst) == 7);
            assert_eq!(prefilled.load(Ordering::SeqCst), 2, "A's and B's prompts");
            assert_eq!(
                relock(&state.shared.stats).n_cancelled_before_prompt_total,
                1,
                "the request that left was cancelled before the engine moved"
            );

            let reset_in_move = complete(addr, "ghgh", 1);
            queued("the request that resets queued");
            release(7, "A's fourth step");
            wait_for("A parked for the move", || {
                calls.load(Ordering::SeqCst) == 8
            });
            reset(reset_in_move);
            wait_for("its connection's release", || {
                state.shared.answers.in_flight() == 1
            });
            // Permits ahead for the rest: the move, the reset request's
            // prompt, A's remaining steps and one more request.
            for _ in 0..64 {
                permit.send(()).expect("release the rest");
            }
            assert_eq!(answer(a)["tokens_predicted"], 40);
            assert_eq!(answer(complete(addr, "ijij", 1))["content"], "i");
        }

        #[test]
        fn lifecycle_stalled_writers_release_their_connection() {
            for streaming in [false, true] {
                let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
                let mut client =
                    TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
                let (server, _) = listener.accept().expect("accept");
                let live = Arc::new(AtomicUsize::new(0));
                let (done, completed) = mpsc::channel();
                let serve = Arc::new(move |server| {
                    super::super::keep_alive(
                        server,
                        &super::super::Answers::default(),
                        &crate::flag::ApiKeys::default(),
                        |req, w| {
                            assert!(
                                w.write_timeout()? == Some(Duration::from_secs(10)),
                                "response writes must time out after ten seconds"
                            );
                            w.set_write_timeout(Some(Duration::from_millis(100)))?;
                            let body = vec![b'x'; 16 << 20];
                            let result = if streaming {
                                super::super::http::EventStream::start(
                                    w,
                                    req,
                                    200,
                                    "text/event-stream",
                                )?
                                .send(&body)
                            } else {
                                super::super::http::respond(w, req, 200, "text/plain", &[], &body)
                            };
                            let kind = result.expect_err("the reader made no progress").kind();
                            assert!(
                                matches!(kind, io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut),
                                "{kind:?}"
                            );
                            Ok(false)
                        },
                    );
                    done.send(()).expect("done");
                });
                super::super::admit(&serve, &live, &Arc::new(AtomicUsize::new(0)), 0, server);
                client
                    .write_all(b"GET /health HTTP/1.0\r\n\r\n")
                    .expect("request");
                let result = completed.recv_timeout(Duration::from_secs(3));
                drop(client);
                result.expect("the stalled writer did not finish");
                wait_for("the writer's connection release", || {
                    live.load(Ordering::SeqCst) == 0
                });
            }
        }

        #[test]
        fn lifecycle_refusal_does_not_block_admission() {
            use super::super::MAX_CONNECTIONS;

            for (held, rejecting) in [
                (MAX_CONNECTIONS, 0),
                (MAX_CONNECTIONS, 1),
                (MAX_CONNECTIONS - 1, 1),
            ] {
                let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
                let mut client =
                    TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
                client
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .expect("timeout");
                let (server, _) = listener.accept().expect("accept");
                let live = Arc::new(AtomicUsize::new(held));
                let refusing = Arc::new(AtomicUsize::new(rejecting));
                let serve = Arc::new(|mut stream: TcpStream| {
                    stream.write_all(b"admitted").expect("answer")
                });
                let start = Instant::now();
                super::super::admit(&serve, &live, &refusing, 0, server);
                assert!(
                    start.elapsed() < super::super::REFUSE_DRAIN / 2,
                    "accept waited {:?} for a refusal",
                    start.elapsed()
                );
                let mut answer = String::new();
                client.read_to_string(&mut answer).expect("answer");
                if held < MAX_CONNECTIONS {
                    assert_eq!(answer, "admitted", "a refusal blocked a free place");
                } else if rejecting == 0 {
                    assert!(answer.starts_with("HTTP/1.1 503"), "{answer}");
                    assert!(answer.contains("Retry-After: 1\r\n"), "{answer}");
                } else {
                    assert!(answer.is_empty(), "an extra refusal was admitted: {answer}");
                }
                drop(client);
                wait_for("the connection's permit release", || {
                    live.load(Ordering::SeqCst) == held
                        && refusing.load(Ordering::SeqCst) == rejecting
                });
            }
        }

        /// Polls `ok` until it holds; a test that waits on a thread that never
        /// answers fails by name instead of hanging.
        fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
            let until = Instant::now() + Duration::from_secs(10);
            while !ok() {
                assert!(Instant::now() < until, "timed out waiting for {what}");
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        /// Whether `s`, which has nothing to read, is in blocking mode: a peek
        /// waits out the read timeout there and fails at once in non-blocking
        /// mode. A correct socket never answers early, so this does not flake.
        fn blocking(s: &TcpStream) -> bool {
            let wait = Duration::from_millis(100);
            s.set_read_timeout(Some(wait)).expect("timeout");
            let t = Instant::now();
            let e = s.peek(&mut [0u8; 1]).expect_err("nothing to read");
            assert_eq!(e.kind(), io::ErrorKind::WouldBlock, "{e}");
            t.elapsed() >= wait / 2
        }

        /// The probe over a real socket pair: a quiet peer and a peer with
        /// bytes pending are clients (the bytes are still there to read, the
        /// next request's among them), read EOF permits an answer, and the
        /// socket stays blocking after a probe that found a client — the
        /// connection's reader shares it.
        #[test]
        fn the_probe_tells_a_client_from_a_gone_one_and_leaves_the_socket_blocking() {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let mut client =
                TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
            let (mut server, _) = listener.accept().expect("accept");
            server
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("timeout");

            probe_client(&server).expect("a quiet client is present");
            assert!(blocking(&server), "blocking after a quiet probe");

            client.write_all(b"hi").expect("write");
            server.peek(&mut [0u8; 1]).expect("the bytes arrive");
            probe_client(&server).expect("pending bytes are a client");
            let mut got = [0u8; 2];
            server.read_exact(&mut got).expect("read");
            assert_eq!(&got, b"hi", "the probe consumed nothing");
            assert!(blocking(&server), "blocking after a probe that saw bytes");

            drop(client);
            assert_eq!(server.read(&mut got).expect("the end of the stream"), 0);
            // PIN(2026-10-23): EOF permits a response to a write-side shutdown; reset is checked separately.
            probe_client(&server).expect("EOF alone does not close the response side");
        }

        /// A whole-answer request whose client resets its socket after the
        /// generation started ends at the next event and frees its slot. The
        /// engine is held at a gate and given one step at a time: a
        /// generation that is not cancelled takes every step the test gives,
        /// to a context the mock never ends, and the slot is still running.
        #[cfg(unix)]
        #[test]
        fn a_whole_answer_whose_client_closed_ends_at_the_next_event() {
            let calls = Arc::new(AtomicUsize::new(0));
            let (permit, permits) = mpsc::channel();
            let engine = Gated {
                inner: MockEngine::new(1 << 16),
                calls: Arc::clone(&calls),
                permits,
                prefilled: Arc::new(AtomicUsize::new(0)),
            };
            let (addr, state, _ended) =
                testserve::spawn(Box::new(engine), testserve::mock_config());
            let free = || relock(&state.shared.board).slots()[0].state == Use::Free;
            let called = || calls.load(Ordering::SeqCst);

            // The mock continues "abab" as `a`, `b`, `a`, `b` and never ends.
            let client = testserve::send_request(
                addr,
                "POST",
                "/completion",
                &[],
                r#"{"prompt":"abab","temperature":0}"#,
            );
            wait_for("the generation's first step", || called() >= 1);
            assert!(!free(), "the request holds the slot");
            reset(client);

            wait_for("the HTTP handler to observe the reset", || {
                state.shared.answers.in_flight() == 0
            });
            permit.send(()).expect("the engine holds its gate");
            wait_for("the slot's release at the next event", free);
            // PIN(2026-10-24): reset detection precedes release, so only the already blocked engine call may run.
            assert_eq!(called(), 1, "the request took another engine step");
        }
    }

    /// The orderly stop's wait on the engine thread — what it waits for (the
    /// loop's end, between engine calls, every request the thread held
    /// answered and the engine closed by then, and the answers in
    /// flight written), and that the thread leaves the engine to the process
    /// exit — pinned at a small scale: [`ENGINE_STOP`]'s policy without
    /// spending its seconds. The servers start the way `run` starts them
    /// ([`Server::start`]); the engines are mocks owned by these gates alone.
    mod stop_wait {
        use std::io::Read;
        use std::net::{SocketAddr, TcpStream};
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex, mpsc};
        use std::time::{Duration, Instant};

        use serde_json::{Value, json};

        use crate::api::{
            FATAL_LINGER, Server, ServerConfig, Shared, State, Stop, Waited, exit_shutdown,
            refused, relock, stop_line, stopping, stopping_message, testserve,
        };
        use crate::genloop::Slot;
        use crate::sched::{Reserve, Use};
        use crate::swap::Turn;
        use crate::worker::{Acted, Action};
        use crate::{
            Engine, EngineError, MockEngine, MockTokenizer, Park, SlotConfig, SwapEngine, Tokenizer,
        };

        /// The waits' bound in these gates: past it, a call the stop arrives
        /// in is reported.
        const BOUND: Duration = Duration::from_millis(400);

        /// How long the witness takes to close: a close made after
        /// the loop's end is published still runs when the stop's wait wakes.
        const CLOSING: Duration = Duration::from_millis(100);

        /// An engine whose Drop notes that it ran, and which counts its
        /// closes, each [`CLOSING`] long.
        struct Witness {
            dropped: Arc<AtomicBool>,
            closed: Arc<AtomicUsize>,
        }

        impl Engine for Witness {
            fn tokenizer(&self) -> Arc<dyn Tokenizer> {
                Arc::new(MockTokenizer)
            }
            fn prefill(&mut self, _ids: &[u32]) -> Result<(), crate::EngineError> {
                Ok(())
            }
            fn next(
                &mut self,
                _last: u32,
                _out: Option<&mut [f32]>,
            ) -> Result<u32, crate::EngineError> {
                Ok(7)
            }
            fn reset(&mut self) -> Result<(), crate::EngineError> {
                Ok(())
            }
            fn ctx_max(&self) -> usize {
                4096
            }
            fn describe(&self) -> String {
                "the witness mock".to_owned()
            }
            fn close(&mut self) -> Result<(), crate::EngineError> {
                std::thread::sleep(CLOSING);
                self.closed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        impl Drop for Witness {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::SeqCst);
            }
        }

        /// An engine whose prompt call blocks until its `go` channel opens:
        /// the call a stop arrives inside.
        struct Blocked {
            go: mpsc::Receiver<()>,
        }

        impl Engine for Blocked {
            fn tokenizer(&self) -> Arc<dyn Tokenizer> {
                Arc::new(MockTokenizer)
            }
            fn prefill(&mut self, _ids: &[u32]) -> Result<(), crate::EngineError> {
                let _ = self.go.recv();
                Ok(())
            }
            fn next(
                &mut self,
                _last: u32,
                _out: Option<&mut [f32]>,
            ) -> Result<u32, crate::EngineError> {
                Ok(7)
            }
            fn reset(&mut self) -> Result<(), crate::EngineError> {
                Ok(())
            }
            fn ctx_max(&self) -> usize {
                4096
            }
            fn describe(&self) -> String {
                "the blocked mock".to_owned()
            }
        }

        /// A server on `engine`, its engine thread started the way `run`
        /// starts it; the shared its stop waits on. Nothing is asked of the
        /// listener, and the engine thread ends detached, as there.
        fn started(engine: Box<dyn Engine>) -> Arc<Shared> {
            let config = ServerConfig {
                model_alias: "mock".to_owned(),
                model_path: "mock.gguf".to_owned(),
                chat_preparation: None,
                chat_template: concat!(
                    "{%- for message in messages %}",
                    "{{- '<' + message.role + '>\\n' + message.content }}",
                    "{%- endfor %}",
                    "{%- if add_generation_prompt %}{{- '<assistant>\\n' }}{%- endif %}",
                )
                .to_owned(),
                sampler: None,
                fatal_linger: FATAL_LINGER,
                slot_save_path: None,
                api_keys: crate::flag::ApiKeys::default(),
            };
            let server = Server::bind("127.0.0.1:0", engine, config).expect("bind");
            let (state, _ended) = server.start().expect("start");
            Arc::clone(&state.shared)
        }

        /// Whatever ends the engine thread's loop ends the server, and the
        /// process exit reclaims the engine: the thread never drops it, so
        /// no driver call of the engine's Drop runs beside the exit's
        /// teardown of the driver's libraries. The thread closes the engine
        /// once, before the loop's end is published, so the stop's wait
        /// covers it. Pinned for each way
        /// the loop ends: the stop, which an idle thread leaves at once (the
        /// wait wakes on the loop's own leaving, not on the bound), an engine
        /// failure and a panic; and for the stop of slots that take the
        /// engine in turns, whose [`SwapEngine`] passes the close on.
        // PIN(2026-10-23): the engine is no longer dropped after the loop; the check that the
        // engine thread finished its Drop became the check that the Drop never runs.
        #[test]
        fn the_engine_is_left_to_the_process_exit() {
            let mut dropped_after = Vec::new();
            let mut closes = Vec::new();
            for end in ["stop", "failure", "panic", "stop of turns"] {
                let dropped = Arc::new(AtomicBool::new(false));
                let closed = Arc::new(AtomicUsize::new(0));
                let witness = Box::new(Witness {
                    dropped: Arc::clone(&dropped),
                    closed: Arc::clone(&closed),
                });
                let sh = started(if end == "stop of turns" {
                    Box::new(SwapEngine::new(witness, 2, Park::Ids).expect("the swap engine"))
                } else {
                    witness
                });
                if end.starts_with("stop") {
                    sh.begin_stop(&Stop::Sigint);
                } else {
                    let fails = end == "failure";
                    let action = Action {
                        run: Box::new(move |_: &mut Slot| {
                            if !fails {
                                panic!("the test's panic on the engine thread");
                            }
                            Acted {
                                failure: Some(EngineError("the test's failure".to_owned())),
                                reply: Box::new(|| ()),
                            }
                        }),
                        drops: false,
                    };
                    relock(&sh.board)
                        .reserve(Reserve::One(0), action)
                        .expect("the slot is free");
                    sh.work.notify_one();
                }
                assert!(sh.wait_loop_end(BOUND), "{end}: the loop left");
                let at_end = closed.load(Ordering::SeqCst);
                std::thread::sleep(BOUND);
                if dropped.load(Ordering::SeqCst) {
                    dropped_after.push(end);
                }
                closes.push((end, at_end, closed.load(Ordering::SeqCst)));
            }
            assert!(
                dropped_after.is_empty(),
                "the engine thread dropped its engine after its loop ended by {dropped_after:?}"
            );
            assert!(
                closes
                    .iter()
                    .all(|&(_, at_end, after)| at_end == 1 && after == 1),
                "the engine is closed once before the loop's end is published; \
                 (loop ended by, closes at its end, closes after) {closes:?}"
            );
        }

        /// The stop is stored and signalled under the board lock, the lock the
        /// engine thread reads it and waits on its condvar with. Stored
        /// outside the lock, it could land between the thread's read and its
        /// wait and be lost, the stop then waiting out [`ENGINE_STOP`]. An
        /// action holds the board lock on the engine thread for a window while
        /// the stop begins from this thread: it must not be published within
        /// the window, and is published once the lock is released, the engine
        /// thread then leaving its loop.
        #[test]
        fn the_stop_is_published_under_the_board_lock() {
            let sh = started(Box::new(Witness {
                dropped: Arc::new(AtomicBool::new(false)),
                closed: Arc::new(AtomicUsize::new(0)),
            }));
            let (locked, door) = mpsc::channel();
            let (seen, verdict) = mpsc::channel();
            let held = Arc::clone(&sh);
            let action = Action {
                run: Box::new(move |_: &mut Slot| {
                    let board = relock(&held.board);
                    let _ = locked.send(());
                    std::thread::sleep(Duration::from_millis(100));
                    let _ = seen.send(held.stopped());
                    drop(board);
                    Acted {
                        failure: None,
                        reply: Box::new(|| ()),
                    }
                }),
                drops: false,
            };
            relock(&sh.board)
                .reserve(Reserve::One(0), action)
                .expect("the slot is free");
            sh.work.notify_one();
            door.recv_timeout(Duration::from_secs(5))
                .expect("the engine thread holds the board lock");
            assert!(sh.begin_stop(&Stop::Sigint), "first cause");
            let stored = verdict
                .recv_timeout(Duration::from_secs(5))
                .expect("the engine thread read the stop");
            assert!(
                !stored,
                "the stop was stored with the board lock held: a check and wait of the \
                 engine thread under that lock can miss it"
            );
            assert!(
                sh.stopped(),
                "the stop was published once the lock was free"
            );
            assert!(
                sh.wait_loop_end(BOUND),
                "the engine thread left its loop on the stop"
            );
        }

        /// The call the stop arrives inside is waited for: the loop cannot
        /// leave until its call ends, so the bound runs out and the stop's
        /// line reports it; released, the loop leaves at its next check
        /// between calls.
        #[test]
        fn the_stop_waits_out_the_call_it_arrives_in() {
            let (go, gate) = mpsc::channel();
            let (entered, door) = mpsc::channel();
            let sh = started(Box::new(Blocked { go: gate }));
            // The action whose run makes the prompt call the stop arrives in.
            let action = Action {
                run: Box::new(move |slot: &mut Slot| {
                    let _ = entered.send(());
                    let called = slot.engine.prefill(&[1]);
                    Acted {
                        failure: called.err(),
                        reply: Box::new(|| ()),
                    }
                }),
                drops: false,
            };
            relock(&sh.board)
                .reserve(Reserve::One(0), action)
                .expect("the slot is free");
            sh.work.notify_one();
            door.recv_timeout(Duration::from_secs(5))
                .expect("the engine thread began its call");
            sh.begin_stop(&Stop::Sigint);
            let waited = sh.wait_stop(BOUND);
            assert_eq!(
                waited,
                Waited::Loop,
                "the loop was inside the call the stop arrived in"
            );
            assert_eq!(
                stop_line(&Stop::Sigint, waited),
                "bloomery-serve: shutdown (SIGINT); the engine thread did not end within 5 s"
            );
            drop(go);
            let until = Instant::now() + Duration::from_secs(5);
            while !sh.wait_loop_end(Duration::from_millis(50)) && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(
                sh.wait_loop_end(Duration::from_millis(50)),
                "the loop left at the first check after its call"
            );
        }

        /// How long a halted request's answer may take to reach its client.
        const READ: Duration = Duration::from_secs(2);
        /// The longest a wait for the engine thread waits: a test that fails
        /// ends without hanging.
        const HOLD: Duration = Duration::from_secs(10);
        /// The quantum of the planned prompts.
        const Q: usize = 4;
        /// 21 ids, so a prompt call of 20: five quanta.
        const LONG: &str = "bcdefghijklmnopqrstub";

        /// An engine call as [`Paced`] logs it, on the slot it was made on.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Call {
            Pre(usize),
            Next(usize),
            Reset(usize),
        }

        /// The mock as built (its slots, its quantum), every call logged in
        /// order, each call in `hold` waiting once logged for a permit the
        /// test sends. Its Drop notes that it ran.
        struct Paced {
            inner: MockEngine,
            cur: usize,
            log: Arc<Mutex<Vec<Call>>>,
            hold: Vec<Call>,
            permits: mpsc::Receiver<()>,
            dropped: Arc<AtomicBool>,
        }

        impl Paced {
            fn call(&mut self, c: Call) -> Result<(), EngineError> {
                relock(&self.log).push(c);
                if self.hold.contains(&c) {
                    self.permits
                        .recv()
                        .map_err(|_| EngineError("the test's gate closed".to_owned()))?;
                }
                Ok(())
            }
        }

        impl Engine for Paced {
            fn tokenizer(&self) -> Arc<dyn Tokenizer> {
                self.inner.tokenizer()
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
                self.call(Call::Pre(self.cur))?;
                self.inner.prefill(ids)
            }
            fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
                self.call(Call::Next(self.cur))?;
                self.inner.next(last, out)
            }
            fn reset(&mut self) -> Result<(), EngineError> {
                self.call(Call::Reset(self.cur))?;
                self.inner.reset()
            }
            fn keepable(&self, n: usize) -> usize {
                self.inner.keepable(n)
            }
            fn cut(&mut self, n: usize) -> Result<(), EngineError> {
                self.inner.cut(n)
            }
            fn ctx_max(&self) -> usize {
                self.inner.ctx_max()
            }
            fn describe(&self) -> String {
                "the paced mock".to_owned()
            }
            fn slots(&self) -> usize {
                self.inner.slots()
            }
            fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
                self.inner.select_slot(slot)?;
                self.cur = slot;
                Ok(())
            }
            fn prompt_quantum(&self) -> Option<std::num::NonZeroUsize> {
                self.inner.prompt_quantum()
            }
        }

        impl Drop for Paced {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::SeqCst);
            }
        }

        /// A served [`Paced`] mock and the test's handles on it: its call
        /// log, its permits, and whether it was dropped.
        struct Rig {
            addr: SocketAddr,
            state: Arc<State>,
            log: Arc<Mutex<Vec<Call>>>,
            permit: mpsc::Sender<()>,
            dropped: Arc<AtomicBool>,
        }

        impl Rig {
            /// `inner` paced, the calls in `hold` held, served on `parallel`
            /// slots — its own, or taking it in turns parked as ids
            /// (`turns`) — with slot files in `save`, if any.
            fn new(
                inner: MockEngine,
                hold: &[Call],
                parallel: usize,
                turns: bool,
                save: Option<PathBuf>,
            ) -> Rig {
                let log = Arc::new(Mutex::new(Vec::new()));
                let (permit, permits) = mpsc::channel();
                let dropped = Arc::new(AtomicBool::new(false));
                let paced = Paced {
                    inner,
                    cur: 0,
                    log: Arc::clone(&log),
                    hold: hold.to_vec(),
                    permits,
                    dropped: Arc::clone(&dropped),
                };
                let engine: Box<dyn Engine> = if turns {
                    Box::new(
                        SwapEngine::new(Box::new(paced), parallel, Park::Ids)
                            .unwrap_or_else(|e| panic!("swap engine: {e}")),
                    )
                } else {
                    Box::new(paced)
                };
                let config = ServerConfig {
                    slot_save_path: save,
                    ..testserve::mock_config()
                };
                let slots = SlotConfig {
                    parallel,
                    ..SlotConfig::default()
                };
                let server = Server::bind_with("127.0.0.1:0", engine, config, slots).expect("bind");
                let addr = server.local_addr().expect("addr");
                let (state, _ended) = server.start().expect("start");
                Rig {
                    addr,
                    state,
                    log,
                    permit,
                    dropped,
                }
            }

            fn sh(&self) -> &Shared {
                &self.state.shared
            }

            /// How many times `c` was called.
            fn count(&self, c: Call) -> usize {
                relock(&self.log).iter().filter(|&&x| x == c).count()
            }

            /// Waits until `c` was called `k` times: a held `c` then stands
            /// at its `k`-th call.
            fn reach(&self, c: Call, k: usize) {
                until(&format!("{c:?} #{k}: {:?}", relock(&self.log)), || {
                    self.count(c) >= k
                });
            }

            /// Lets the held call run.
            fn permit(&self) {
                self.permit.send(()).expect("the engine holds its gate");
            }

            /// `body` posted to `path` on a connection of its own, its answer
            /// unread.
            fn post(&self, path: &str, body: &str) -> TcpStream {
                testserve::send_request(self.addr, "POST", path, &[], body)
            }

            /// Waits until `n` requests wait for a slot.
            fn queued(&self, n: usize) {
                until(&format!("{n} requests waiting"), || {
                    relock(&self.sh().board).waiting() == n
                });
            }

            /// Waits until an action holds slot `i`.
            fn reserved(&self, i: usize) {
                until(&format!("an action on slot {i}"), || {
                    relock(&self.sh().board).slots()[i].state == Use::Held
                });
            }
        }

        /// Polls `ok` until it holds; a wait that never ends fails by name.
        fn until(what: &str, mut ok: impl FnMut() -> bool) {
            let deadline = Instant::now() + HOLD;
            while !ok() {
                assert!(Instant::now() < deadline, "timed out waiting for {what}");
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        /// The answer `s` reads to its connection's end, which must come
        /// within [`READ`]: its status and body.
        fn answer(mut s: TcpStream) -> (u16, String) {
            s.set_read_timeout(Some(READ)).expect("timeout");
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                match s.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => raw.extend_from_slice(&buf[..n]),
                    Err(e) => panic!(
                        "the answer did not end within {READ:?} ({e}); so far: {}",
                        String::from_utf8_lossy(&raw)
                    ),
                }
            }
            let raw = String::from_utf8(raw).expect("the answer is UTF-8");
            let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
            let status = head
                .split_whitespace()
                .nth(1)
                .and_then(|c| c.parse().ok())
                .expect("a status line");
            (status, body.to_owned())
        }

        /// What the stop answers a request it ended, the queued requests'
        /// 503 naming its cause: a whole answer's body, a stream's last
        /// event.
        fn stop_answer() -> Value {
            json!({ "error": {
                "code": 503,
                "message": stopping_message("SIGINT"),
                "type": "unavailable_error",
            } })
        }

        fn json_of(body: &str) -> Value {
            serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
        }

        /// The last event of a stream's body.
        fn last_event(body: &str) -> Value {
            let ev = body
                .split("\n\n")
                .filter(|e| !e.is_empty())
                .last()
                .unwrap_or_else(|| panic!("no event: {body}"));
            json_of(
                ev.strip_prefix("data: ")
                    .unwrap_or_else(|| panic!("not a data event: {ev}")),
            )
        }

        /// A directory of its own for a test's slot files.
        fn fresh_dir(name: &str) -> PathBuf {
            let dir = std::env::temp_dir()
                .join(format!("bloomery-stop-wait-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("the slot files' directory");
            dir
        }

        /// Every request the worker holds when the stop arrives is answered
        /// before the loop's end, the engine left to the process exit: a
        /// stream mid-generation ends with the stop's error event and its end,
        /// no `[DONE]` (the end an engine failure gives a stream); a request
        /// between its prompt's calls — a quantum's plan beside the busy
        /// stream — gets the stop's 503, the queued requests' answer. Both
        /// are answers in flight, and the stop's wait ends once they are
        /// written. The stop is read before the next planned call and before
        /// the decode round: neither runs after it. The halted requests'
        /// counters are booked; their slots stay as the stop found them, so
        /// `/slots` and `/health` show what they showed before it.
        #[test]
        fn the_halt_answers_every_request_it_holds_before_the_loop_ends() {
            let rig = Rig::new(
                MockEngine::new(4096).with_slots(2).with_prompt_quantum(Q),
                &[Call::Next(0)],
                2,
                false,
                None,
            );
            // A streams on slot 0: its prompt's step, then its first decode
            // step held, its first token's event sent.
            let a = rig.post(
                "/completion",
                r#"{"prompt":"abab","temperature":0,"stream":true}"#,
            );
            rig.reach(Call::Next(0), 1);
            rig.permit();
            rig.reach(Call::Next(0), 2);
            // B takes slot 1 beside the busy stream: its prompt is planned as
            // five calls, one a round; two run.
            let b = rig.post(
                "/completion",
                &format!(r#"{{"prompt":"{LONG}","temperature":0,"n_predict":4}}"#),
            );
            rig.queued(1);
            rig.permit();
            rig.reach(Call::Next(0), 3);
            rig.permit();
            rig.reach(Call::Next(0), 4);
            assert_eq!(rig.count(Call::Pre(1)), 2, "B is between its calls");
            assert_eq!(rig.sh().answers.in_flight(), 2, "A's and B's answers");

            rig.sh().begin_stop(&Stop::Sigint);
            rig.permit();
            assert_eq!(
                rig.sh().wait_stop(BOUND),
                Waited::Done,
                "the loop left and both answers were written"
            );
            assert!(
                !rig.dropped.load(Ordering::SeqCst),
                "the engine thread dropped its engine after the loop"
            );
            let (status, body) = answer(a);
            assert_eq!(status, 200, "{body}");
            assert!(body.contains(r#""content":"#), "A was generating: {body}");
            assert_eq!(last_event(&body), stop_answer(), "{body}");
            assert!(!body.contains("[DONE]"), "{body}");
            let (status, body) = answer(b);
            assert_eq!(status, 503, "{body}");
            assert_eq!(json_of(&body), stop_answer());

            assert_eq!(
                rig.count(Call::Pre(1)),
                2,
                "a planned call ran after the stop"
            );
            assert_eq!(
                rig.count(Call::Next(0)),
                4,
                "a decode round ran after the stop"
            );
            assert_eq!(
                relock(&rig.sh().stats).n_prompt_total,
                (4 + 2 * Q) as u64,
                "A's prompt and B's two calls are booked"
            );
            let (status, slots) = testserve::roundtrip(rig.addr, "GET", "/slots", &[], "");
            assert_eq!(status, 200, "{slots}");
            let slots = json_of(&slots);
            for s in 0..2 {
                assert_eq!(slots[s]["is_processing"], true, "slot {s}: {slots}");
            }
            let (status, health) = testserve::roundtrip(rig.addr, "GET", "/health", &[], "");
            assert_eq!(status, 200, "{health}");
            assert_eq!(json_of(&health)["slots_processing"], 2, "{health}");
        }

        /// The stop's wait covers the answers in flight: with the loop left
        /// and an answer still being written, the wait runs out on it and the
        /// stop line says so; once it is written, the wait ends and the line
        /// names the cause alone.
        #[test]
        fn the_stop_waits_for_the_answers_in_flight() {
            let sh = started(Box::new(Witness {
                dropped: Arc::new(AtomicBool::new(false)),
                closed: Arc::new(AtomicUsize::new(0)),
            }));
            let writing = sh.answers.answering();
            sh.begin_stop(&Stop::Sigint);
            let waited = sh.wait_stop(BOUND);
            assert_eq!(waited, Waited::Answers(1), "an answer was being written");
            assert_eq!(
                stop_line(&Stop::Sigint, waited),
                "bloomery-serve: shutdown (SIGINT); 1 answer(s) were still being written at \
                 the 5 s bound"
            );
            drop(writing);
            let waited = sh.wait_stop(BOUND);
            assert_eq!(waited, Waited::Done, "the answer was written");
            assert_eq!(
                stop_line(&Stop::Sigint, waited),
                "bloomery-serve: shutdown (SIGINT)"
            );
        }

        /// The stop's exit tears the process down (its exit handlers, the
        /// loaded libraries' destructors) only after a wait that ended: after
        /// one that ran out, a device driver call can still be running, and
        /// the process ends without the teardown. Each way the wait ends
        /// exits a process of its own, which notes whether an exit handler
        /// ran.
        #[test]
        fn the_stop_tears_the_process_down_only_after_a_wait_that_ended() {
            const NAME: &str = "api::tests::stop_wait::\
                                the_stop_tears_the_process_down_only_after_a_wait_that_ended";
            // The fresh process's own argument: a filter no test's name is.
            const FRESH: &str = "a-fresh-process-for-the-stop-exit";
            const TORN: &str = "an exit handler ran";
            extern "C" fn torn() {
                eprintln!("{TORN}");
            }
            if let Some(case) = std::env::args().skip_while(|a| a != FRESH).nth(1) {
                // SAFETY: `torn` is a function of no arguments that lives as
                // long as the process.
                assert_eq!(unsafe { libc::atexit(torn) }, 0, "the exit handler");
                let waited = match case.as_str() {
                    "done" => Waited::Done,
                    "loop" => Waited::Loop,
                    "answers" => Waited::Answers(1),
                    other => panic!("no wait ends as {other}"),
                };
                exit_shutdown(&Stop::Sigterm, |_| waited);
            }
            let exe = std::env::current_exe().expect("the test binary");
            for (case, torn_down) in [("done", true), ("loop", false), ("answers", false)] {
                let out = std::process::Command::new(&exe)
                    .args(["--exact", NAME, FRESH, case, "--nocapture"])
                    .output()
                    .expect("the fresh process");
                let err = String::from_utf8_lossy(&out.stderr);
                assert!(
                    out.status.success() && err.contains("bloomery-serve: shutdown (SIGTERM)"),
                    "{case}: {}\n{err}",
                    out.status
                );
                assert_eq!(err.contains(TORN), torn_down, "{case}: {err}");
            }
        }

        /// The stop is read before each action and each start of a round:
        /// two actions the round took together, the stop arriving inside the
        /// first, leave the second unrun, its client answered with the stop;
        /// two requests the round admitted together, the stop arriving inside
        /// the first's prompt, leave the second unstarted, answered the same.
        /// The round's decode step runs neither time.
        #[test]
        fn the_stop_is_read_before_each_action_and_start_of_a_round() {
            let dir = fresh_dir("actions");
            let rig = Rig::new(
                MockEngine::new(4096).with_slots(3),
                &[Call::Next(0), Call::Reset(1)],
                3,
                false,
                Some(dir.clone()),
            );
            let a = rig.post("/completion", r#"{"prompt":"abab","temperature":0}"#);
            rig.reach(Call::Next(0), 1);
            rig.permit();
            rig.reach(Call::Next(0), 2);
            let first = rig.post("/slots/1?action=erase", "");
            rig.reserved(1);
            let second = rig.post("/slots/2?action=erase", "");
            rig.reserved(2);
            rig.permit();
            rig.reach(Call::Reset(1), 1);
            rig.sh().begin_stop(&Stop::Sigint);
            rig.permit();
            assert_eq!(rig.sh().wait_stop(BOUND), Waited::Done);
            let (status, body) = answer(first);
            assert_eq!(status, 200, "the action the stop arrived in: {body}");
            let (status, body) = answer(second);
            assert_eq!((status, json_of(&body)), (503, stop_answer()), "{body}");
            assert_eq!(
                rig.count(Call::Reset(2)),
                0,
                "the second action ran after the stop"
            );
            assert_eq!(
                rig.count(Call::Next(0)),
                2,
                "a decode round ran after the stop"
            );
            let (status, body) = answer(a);
            assert_eq!((status, json_of(&body)), (503, stop_answer()), "{body}");
            let _ = std::fs::remove_dir_all(&dir);

            let rig = Rig::new(
                MockEngine::new(4096).with_slots(3),
                &[Call::Next(0), Call::Pre(1)],
                3,
                false,
                None,
            );
            let a = rig.post("/completion", r#"{"prompt":"abab","temperature":0}"#);
            rig.reach(Call::Next(0), 1);
            rig.permit();
            rig.reach(Call::Next(0), 2);
            let b = rig.post("/completion", r#"{"prompt":"bcbc","temperature":0}"#);
            rig.queued(1);
            let c = rig.post("/completion", r#"{"prompt":"cdcd","temperature":0}"#);
            rig.queued(2);
            rig.permit();
            rig.reach(Call::Pre(1), 1);
            rig.sh().begin_stop(&Stop::Sigint);
            rig.permit();
            assert_eq!(rig.sh().wait_stop(BOUND), Waited::Done);
            let log = relock(&rig.log).clone();
            assert!(
                !log.contains(&Call::Pre(2)) && !log.contains(&Call::Next(2)),
                "the second start ran after the stop: {log:?}"
            );
            assert_eq!(
                rig.count(Call::Next(0)),
                2,
                "a decode round ran after the stop"
            );
            for (name, s) in [("A", a), ("B", b), ("C", c)] {
                let (status, body) = answer(s);
                assert_eq!(
                    (status, json_of(&body)),
                    (503, stop_answer()),
                    "{name}: {body}"
                );
            }
        }

        /// A request that passed the stop's check before the stop began and
        /// reached the board after the halt killed it is refused with the
        /// stop's 503, as every request the halt ended is, not with an engine
        /// failure's; a board an engine failure killed, no stop begun, still
        /// names the failure.
        #[test]
        fn a_refusal_by_the_board_the_halt_killed_names_the_stop() {
            let action = || Action {
                run: Box::new(|_: &mut Slot| Acted {
                    failure: None,
                    reply: Box::new(|| ()),
                }),
                drops: false,
            };
            let (_addr, state, _ended) =
                testserve::spawn(Box::new(MockEngine::new(64)), testserve::mock_config());
            assert!(state.shared.begin_stop(&Stop::Sigint), "first cause");
            assert!(
                state.shared.wait_loop_end(HOLD),
                "the halt killed the board"
            );
            let r = relock(&state.shared.board)
                .reserve(Reserve::One(0), action())
                .expect_err("the board the halt killed refuses");
            let e = refused(&state, r, "slot 0");
            assert_eq!(
                (e.code, e.kind, e.message),
                (503, "unavailable_error", stopping_message("SIGINT"))
            );

            let (_addr, state, _ended) =
                testserve::spawn(Box::new(MockEngine::new(64)), testserve::mock_config());
            drop(relock(&state.shared.board).kill("the mock failed"));
            let r = relock(&state.shared.board)
                .reserve(Reserve::One(0), action())
                .expect_err("the board an engine failure killed refuses");
            let e = refused(&state, r, "slot 0");
            assert_eq!(
                (e.code, e.kind, e.message),
                (503, "unavailable_error", stopping("the mock failed"))
            );
        }

        /// Under turns the stop is read before the running request's step: a
        /// stop that arrives inside a newcomer's prompt, after the engine
        /// changed hands, ends the loop before that request's first step,
        /// and both requests get the stop's 503.
        #[test]
        fn under_turns_the_stop_is_read_before_the_running_requests_step() {
            let rig = Rig::new(
                MockEngine::new(4096),
                &[Call::Pre(0), Call::Next(0)],
                2,
                true,
                None,
            );
            let a = rig.post("/completion", r#"{"prompt":"abab","temperature":0}"#);
            rig.reach(Call::Pre(0), 1);
            rig.permit();
            rig.reach(Call::Next(0), 1);
            rig.permit();
            rig.reach(Call::Next(0), 2);
            // B preempts A at A's next step: A parks and B's prompt runs.
            let b = rig.post("/completion", r#"{"prompt":"bcbc","temperature":0}"#);
            rig.queued(1);
            rig.permit();
            rig.reach(Call::Pre(0), 2);
            rig.sh().begin_stop(&Stop::Sigint);
            rig.permit();
            // B's prompt's own step, inside the start the stop arrived in.
            rig.reach(Call::Next(0), 3);
            rig.permit();
            assert_eq!(
                rig.sh().wait_stop(BOUND),
                Waited::Done,
                "the loop left: {:?}",
                relock(&rig.log)
            );
            assert_eq!(rig.count(Call::Next(0)), 3, "B stepped after the stop");
            for (name, s) in [("A", a), ("B", b)] {
                let (status, body) = answer(s);
                assert_eq!(
                    (status, json_of(&body)),
                    (503, stop_answer()),
                    "{name}: {body}"
                );
            }
        }

        /// Under turns the halt answers every request it holds too, before
        /// the loop's end and with the engine left to the process exit: the
        /// parked stream ends with the stop's error event, and the running
        /// request, the one waiting for its prompt's turn and the slot action
        /// deferred while they live each get the stop's 503.
        #[test]
        fn under_turns_the_halt_answers_the_parked_the_pending_and_the_deferred() {
            let dir = fresh_dir("turns");
            let rig = Rig::new(
                MockEngine::new(4096),
                &[Call::Next(0)],
                4,
                true,
                Some(dir.clone()),
            );
            let a = rig.post(
                "/completion",
                r#"{"prompt":"abab","temperature":0,"stream":true}"#,
            );
            rig.reach(Call::Next(0), 1);
            rig.permit();
            rig.reach(Call::Next(0), 2);
            // B and C take slots together: B preempts A, whose state parks;
            // C waits for B's turn to end.
            let b = rig.post("/completion", r#"{"prompt":"bcbc","temperature":0}"#);
            rig.queued(1);
            let c = rig.post("/completion", r#"{"prompt":"cdcd","temperature":0}"#);
            rig.queued(2);
            rig.permit();
            rig.reach(Call::Next(0), 3);
            rig.permit();
            rig.reach(Call::Next(0), 4);
            // An action on the free slot waits while requests live.
            let erase = rig.post("/slots/3?action=erase", "");
            rig.reserved(3);
            rig.permit();
            rig.reach(Call::Next(0), 5);
            assert_eq!(
                relock(&rig.sh().turns).clone(),
                Some(vec![Turn::Parked, Turn::Running, Turn::Queued, Turn::Idle]),
                "A parked, B running, C pending"
            );

            rig.sh().begin_stop(&Stop::Sigint);
            rig.permit();
            assert_eq!(rig.sh().wait_stop(BOUND), Waited::Done);
            assert!(
                !rig.dropped.load(Ordering::SeqCst),
                "the engine thread dropped its engine after the loop"
            );
            let (status, body) = answer(a);
            assert_eq!(status, 200, "{body}");
            assert_eq!(last_event(&body), stop_answer(), "{body}");
            for (name, s) in [("B", b), ("C", c), ("the erase", erase)] {
                let (status, body) = answer(s);
                assert_eq!(
                    (status, json_of(&body)),
                    (503, stop_answer()),
                    "{name}: {body}"
                );
            }
            assert_eq!(rig.count(Call::Next(0)), 5, "a step ran after the stop");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    // -------------------------------------------------- chat template rendering

    /// The mock configuration serving `template`'s source.
    fn templated(template: &str) -> super::ServerConfig {
        super::ServerConfig {
            chat_template: template.to_owned(),
            ..super::testserve::mock_config()
        }
    }

    /// `/apply-template`'s status and body for `body`.
    fn applied(addr: SocketAddr, body: &Value) -> (u16, String) {
        roundtrip(addr, "POST", "/apply-template", &body.to_string())
    }

    /// The `prompt` an `/apply-template` answer carries.
    fn applied_prompt(body: &str) -> String {
        serde_json::from_str::<Value>(body).expect("the answer is JSON")["prompt"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }

    /// A `developer` message renders exactly as the same message with its
    /// role set to `system`, on every fixture template
    /// ([`FIXTURE_TEMPLATES`]): llama-server's mapping, which every template
    /// but GPT-OSS's gets. Each template renders the system message — none
    /// drops or refuses it — so the mapped role lands in the prompt's text.
    #[test]
    fn developer_messages_render_as_system_on_every_fixture_template() {
        for (name, template) in FIXTURE_TEMPLATES {
            let (addr, _state, _ended) =
                super::testserve::spawn(Box::new(crate::MockEngine::new(64)), templated(template));
            let with_role = |role: &str| {
                json!({
                    "messages": [
                        { "role": role, "content": "You are terse." },
                        { "role": "user", "content": "Hi" },
                    ]
                })
            };
            let (status, developer) = applied(addr, &with_role("developer"));
            assert_eq!(status, 200, "{name}: {developer}");
            let (status, system) = applied(addr, &with_role("system"));
            assert_eq!(status, 200, "{name}: {system}");
            assert_eq!(developer, system, "{name}: developer renders as system");
            let prompt = applied_prompt(&developer);
            assert!(
                prompt.contains("You are terse."),
                "{name}: the system message renders: {prompt}"
            );
        }
    }

    /// The Bolt template's two renders that differ from the Nano's, on one
    /// conversation (system, user, assistant with `reasoning_content`, user):
    /// a `chat_template_kwargs` `medium_effort` appends
    /// `\n\n{reasoning effort: efficient}` to the last user turn alone, and
    /// the history's reasoning joins as `<think>\n…</think>…` with no newline
    /// before the close (the Nano's template puts one on each side of it).
    /// `truncate_history_thinking: false` keeps the history's reasoning at
    /// all: both templates default it to truncating the turns before the last
    /// user turn, and serve passes each kwarg through unchanged.
    #[test]
    fn the_bolt_template_renders_medium_effort_and_its_reasoning_join() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            templated(include_str!(
                "../tests/fixtures/nemotron-bolt-chat-template.jinja"
            )),
        );
        let body = json!({
            "messages": [
                { "role": "system", "content": "You are terse." },
                { "role": "user", "content": "First." },
                { "role": "assistant", "content": "Hi", "reasoning_content": "Thought." },
                { "role": "user", "content": "Second." },
            ],
            "chat_template_kwargs": {
                "medium_effort": true,
                "truncate_history_thinking": false,
            },
        });
        let (status, answer) = applied(addr, &body);
        assert_eq!(status, 200, "{answer}");
        let prompt = applied_prompt(&answer);
        assert_eq!(
            prompt.matches("{reasoning effort: efficient}").count(),
            1,
            "the effort suffix lands on one turn: {prompt}"
        );
        assert!(
            prompt.contains("<|im_start|>user\nFirst.<|im_end|>\n"),
            "the earlier user turn carries no effort suffix: {prompt}"
        );
        assert!(
            prompt
                .contains("<|im_start|>user\nSecond.\n\n{reasoning effort: efficient}<|im_end|>\n"),
            "the last user turn ends with the effort suffix: {prompt}"
        );
        assert!(
            prompt.contains("<|im_start|>assistant\n<think>\nThought.</think>Hi<|im_end|>\n"),
            "the history's reasoning joins with no newline before </think>: {prompt}"
        );
    }

    /// A template whose source carries `<|channel|>` — GPT-OSS's marker —
    /// keeps the `developer` role: the mapping is every template's but its.
    #[test]
    fn a_channel_template_keeps_the_developer_role() {
        let template = concat!(
            "{%- for m in messages %}",
            "{%- if m.role == 'developer' %}{{- '<|channel|>developer|>' ~ m.content }}",
            "{%- elif m.role == 'system' %}{{- 'system|' ~ m.content }}",
            "{%- else %}{{- m.role ~ '|' ~ m.content }}",
            "{%- endif %}",
            "{%- endfor %}",
        );
        let (addr, _state, _ended) =
            super::testserve::spawn(Box::new(crate::MockEngine::new(64)), templated(template));
        let with_role = |role: &str| {
            json!({
                "messages": [
                    { "role": role, "content": "You are terse." },
                    { "role": "user", "content": "Hi" },
                ]
            })
        };
        let (status, developer) = applied(addr, &with_role("developer"));
        assert_eq!(status, 200, "{developer}");
        assert_eq!(
            applied_prompt(&developer),
            "<|channel|>developer|>You are terse.user|Hi"
        );
        let (status, system) = applied(addr, &with_role("system"));
        assert_eq!(status, 200, "{system}");
        assert_eq!(applied_prompt(&system), "system|You are terse.user|Hi");
    }

    /// `reasoning_effort: "none"` on Qwen3.8's template turns thinking off:
    /// it renders as `chat_template_kwargs: {"enable_thinking": false}` does
    /// — the prompt ends with the closed empty span — it wins over a kwargs
    /// `enable_thinking: true`, and it erases a kwargs `reasoning_effort`.
    /// Another effort reaches the template as today, in its own words.
    #[test]
    fn reasoning_effort_none_turns_thinking_off_on_qwen38() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            templated(include_str!("../tests/fixtures/qwen38-chat-template.jinja")),
        );
        let messages = || json!([{ "role": "user", "content": "Hi" }]);
        let (status, off) = applied(
            addr,
            &json!({ "messages": messages(), "chat_template_kwargs": { "enable_thinking": false } }),
        );
        assert_eq!(status, 200, "{off}");
        for body in [
            json!({ "messages": messages(), "reasoning_effort": "none" }),
            json!({
                "messages": messages(),
                "reasoning_effort": "none",
                "chat_template_kwargs": { "enable_thinking": true },
            }),
            json!({
                "messages": messages(),
                "reasoning_effort": "none",
                "chat_template_kwargs": { "reasoning_effort": "high" },
            }),
        ] {
            let (status, prompt) = applied(addr, &body);
            assert_eq!(status, 200, "{prompt}");
            assert_eq!(prompt, off, "none turns thinking off");
            let p = applied_prompt(&prompt);
            assert!(
                p.ends_with("<think>\n\n</think>\n\n"),
                "the span is closed: {p}"
            );
        }
        let (status, high) = applied(
            addr,
            &json!({ "messages": messages(), "reasoning_effort": "high" }),
        );
        assert_eq!(status, 200, "{high}");
        let p = applied_prompt(&high);
        assert!(
            p.contains("Reasoning effort is set to xhigh."),
            "another effort reaches the template: {p}"
        );
        assert!(p.ends_with("<think>\n"), "its span stays open: {p}");
    }

    /// The official V4.1 encoding's prompts: DeepSeek's own encoder renders
    /// them (the fixture's `expected` text, generated by the recipe's
    /// `DeepseekV41Encoding`), and the official template renders the same
    /// text for the same request.
    ///
    /// A case the chat API cannot express is named in the fixture as such,
    /// and the test holds that its text is one the official encoding writes
    /// and ours does not.
    #[test]
    fn the_official_v41_template_renders_the_recipe_prompts() {
        /// The first byte at which two prompts differ, and the line each
        /// holds there.
        fn difference(expected: &str, got: &str) -> String {
            let at = expected
                .char_indices()
                .zip(got.chars())
                .find(|((_, a), b)| a != b)
                .map_or_else(|| expected.len().min(got.len()), |((at, _), _)| at);
            fn line(text: &str, at: usize) -> &str {
                let start = text[..at].rfind('\n').map_or(0, |at| at + 1);
                let end = text[at..].find('\n').map_or(text.len(), |i| at + i);
                &text[start..end]
            }
            format!(
                "first difference at byte {at}\n  expected: {}\n  got:      {}",
                line(expected, at),
                line(got, at)
            )
        }
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../tests/fixtures/v41-official/cases.json"))
                .expect("the fixture is JSON");
        let mut config = templated(crate::v41::TEMPLATE);
        config.chat_preparation = Some(crate::v41::prepare);
        let (addr, _state, _ended) =
            super::testserve::spawn(Box::new(crate::MockEngine::new(64)), config);
        let mut not_expressible = Vec::new();
        for case in &cases {
            let name = case["name"].as_str().expect("a case has a name");
            let expected = case["expected"].as_str().expect("a case has its text");
            let (status, answer) = applied(addr, &case["request"]);
            assert_eq!(status, 200, "{name}: {answer}");
            let prompt = applied_prompt(&answer);
            if case["expressible"].as_bool().unwrap_or(true) {
                assert_eq!(
                    prompt,
                    expected,
                    "{name}: {}",
                    difference(expected, &prompt)
                );
            } else {
                assert_ne!(
                    prompt, expected,
                    "{name}: the case is not expressible, and renders the official text"
                );
                not_expressible.push(name);
            }
        }
        assert_eq!(not_expressible, ["prefix-continuation"]);
    }

    /// The V4.1 preparation refuses a malformed tool call by name: a
    /// `tool_calls` that is not an array and a call whose `function` is not
    /// an object. The client gets a 400 that names the field, on the render
    /// path and on the chat path alike.
    #[test]
    fn the_v41_preparation_refuses_a_malformed_tool_call_by_name() {
        let mut config = templated(crate::v41::TEMPLATE);
        config.chat_preparation = Some(crate::v41::prepare);
        let (addr, _state, _ended) =
            super::testserve::spawn(Box::new(crate::MockEngine::new(64)), config);
        let cases = [
            (
                json!({"messages": [{"role": "assistant", "tool_calls": [{"function": 1}]}]}),
                "messages[0].tool_calls[0].function",
            ),
            (
                json!({"messages": [{"role": "assistant", "tool_calls": {"function": {}}}]}),
                "messages[0].tool_calls",
            ),
        ];
        for (body, named) in cases {
            for path in ["/apply-template", "/v1/chat/completions"] {
                let (status, answer) = roundtrip(addr, "POST", path, &body.to_string());
                assert_eq!(status, 400, "{path}: {answer}");
                assert!(answer.contains(named), "{path}: {answer}");
            }
        }
    }

    /// The official V4.1 template refuses a role it does not encode, by
    /// name: the client gets a 400 that names the role, not a prompt that
    /// silently drops the message.
    #[test]
    fn the_v41_template_refuses_a_role_it_cannot_encode() {
        let mut config = templated(crate::v41::TEMPLATE);
        config.chat_preparation = Some(crate::v41::prepare);
        let (addr, _state, _ended) =
            super::testserve::spawn(Box::new(crate::MockEngine::new(64)), config);
        let body = json!({"messages": [{"role": "critic", "content": "Tighten it."}]});
        for path in ["/apply-template", "/v1/chat/completions"] {
            let (status, answer) = roundtrip(addr, "POST", path, &body.to_string());
            assert_eq!(status, 400, "{path}: {answer}");
            assert!(
                answer.contains("unsupported message role: critic"),
                "{path}: {answer}"
            );
        }
    }

    /// A server with no request preparation renders a request that carries a
    /// tool choice and a response format as it did before the official
    /// encoding: the two reach no template, so the response format is not
    /// written and no tool call is forced. The tools themselves reach the
    /// template, which every family's does.
    #[test]
    fn a_server_without_a_preparation_ignores_the_tool_choice_and_the_response_format() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            templated(crate::v41::TEMPLATE),
        );
        let body = json!({
            "messages": [{ "role": "user", "content": "Hi" }],
            "tools": [{
                "type": "function",
                "function": { "name": "f", "parameters": { "type": "object" } },
            }],
            "tool_choice": "required",
            "response_format": { "type": "json_object" },
        });
        let (status, answer) = applied(addr, &body);
        assert_eq!(status, 200, "{answer}");
        let prompt = applied_prompt(&answer);
        assert!(
            prompt.contains("### Available Tool Schemas"),
            "the tools reach the template: {prompt}"
        );
        assert!(
            !prompt.contains("## Response Format:"),
            "the response format is not written: {prompt}"
        );
        assert!(
            !prompt.ends_with("\n\n<｜DSML｜ calls>\n"),
            "no tool call is forced: {prompt}"
        );
    }

    /// The official V4.1 template teaches the spaced DSML markup, the one a
    /// V4.1 model writes: the block is ` calls`, the call is ` invoke` and the
    /// parameter is ` parameter`.
    #[test]
    fn the_official_v41_template_teaches_the_spaced_dsml_markup() {
        let parsed = ChatTemplate::parse(crate::v41::TEMPLATE).expect("the template parses");
        assert_eq!(
            crate::dsml::ToolFormat::of_chat_template(&parsed),
            crate::dsml::ToolFormat::Dsml(crate::dsml::Spelling::Spaced)
        );
    }

    /// Any `reasoning_effort` but `"none"` asks for reasoning: on V4.1's
    /// template, which renders no thinking until `enable_thinking` is set,
    /// `low`, `medium` and `high` open the span as the kwargs `true` does, `max`
    /// adds the template's maximum-effort header, an explicit kwargs
    /// `enable_thinking` wins either way, and `"none"` is the template's
    /// default. A template that thinks by default (Qwen3's) renders as it did.
    #[test]
    fn reasoning_effort_asks_for_thinking_on_a_template_that_is_off_by_default() {
        let prompt = |template: &str, extra: Value| {
            let (addr, _state, _ended) =
                super::testserve::spawn(Box::new(crate::MockEngine::new(64)), templated(template));
            let mut body = json!({ "messages": [{ "role": "user", "content": "Hi" }] });
            body.as_object_mut()
                .expect("an object")
                .extend(extra.as_object().expect("an object").clone());
            let (status, answer) = applied(addr, &body);
            assert_eq!(status, 200, "{answer}");
            applied_prompt(&answer)
        };
        let v41 = include_str!("../tests/fixtures/v41-chat-template.jinja");
        let off = prompt(v41, json!({}));
        let on = prompt(
            v41,
            json!({ "chat_template_kwargs": { "enable_thinking": true } }),
        );
        assert!(
            off.ends_with("<｜Assistant｜></think>"),
            "off by default: {off}"
        );
        assert!(
            on.ends_with("<｜Assistant｜><think>"),
            "the switch opens the span: {on}"
        );
        for effort in ["low", "medium", "high"] {
            let got = prompt(v41, json!({ "reasoning_effort": effort }));
            assert_eq!(got, on, "{effort} is thinking at the default effort");
        }
        let max = prompt(v41, json!({ "reasoning_effort": "max" }));
        assert!(max.ends_with("<｜Assistant｜><think>"), "max thinks: {max}");
        assert!(
            max.contains("Reasoning Effort: Absolute maximum"),
            "max reaches the template: {max}"
        );
        assert_eq!(prompt(v41, json!({ "reasoning_effort": "none" })), off);
        for effort in ["high", "max"] {
            let got = prompt(
                v41,
                json!({
                    "reasoning_effort": effort,
                    "chat_template_kwargs": { "enable_thinking": false },
                }),
            );
            assert_eq!(got, off, "{effort}: the explicit switch wins");
        }
        let qwen3 = include_str!("../tests/fixtures/qwen3-chat-template.jinja");
        assert_eq!(
            prompt(qwen3, json!({ "reasoning_effort": "high" })),
            prompt(qwen3, json!({})),
            "a template that thinks by default is asked for nothing"
        );
    }

    /// A `reasoning_effort` outside OpenAI's set is a 400 naming the field, the
    /// value and the accepted ones, on every route that carries it (chat,
    /// `/apply-template`, the Responses API's `reasoning.effort`); every
    /// accepted word renders.
    #[test]
    fn reasoning_effort_outside_openais_set_is_a_named_400() {
        let v41 = include_str!("../tests/fixtures/v41-chat-template.jinja");
        let (addr, _state, _ended) =
            super::testserve::spawn(Box::new(crate::MockEngine::new(64)), templated(v41));
        let accepted = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
        for effort in accepted {
            let (status, answer) = applied(
                addr,
                &json!({ "messages": [{ "role": "user", "content": "Hi" }], "reasoning_effort": effort }),
            );
            assert_eq!(status, 200, "{effort}: {answer}");
        }
        for effort in ["banana", "", "HIGH", "extreme"] {
            let chat = json!({ "messages": [{ "role": "user", "content": "Hi" }], "reasoning_effort": effort });
            for path in ["/v1/chat/completions", "/apply-template"] {
                let (status, answer) = roundtrip(addr, "POST", path, &chat.to_string());
                assert_eq!(status, 400, "{path} {effort:?}: {answer}");
                let message =
                    serde_json::from_str::<Value>(&answer).expect("JSON")["error"]["message"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned();
                assert!(
                    message.contains("\"reasoning_effort\"")
                        && message.contains(&format!("{effort:?}"))
                        && message.contains(&accepted.join(", ")),
                    "{path} {effort:?}: {message}"
                );
            }
            let (status, answer) = roundtrip(
                addr,
                "POST",
                "/v1/responses",
                &json!({ "input": "Hi", "reasoning": { "effort": effort } }).to_string(),
            );
            assert_eq!(status, 400, "responses {effort:?}: {answer}");
            let message = serde_json::from_str::<Value>(&answer).expect("JSON")["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            assert!(message.contains("\"reasoning.effort\""), "{message}");
        }
    }

    /// `reasoning_effort: "none"` on GLM-5's template, which has no thinking
    /// switch (its generation prompt always opens `<think>`) and an effort
    /// header that takes `low` or `high`, is sent as `"low"`, the template's
    /// lowest effort: it renders as `"low"` does, a kwargs `reasoning_effort`
    /// beside it is replaced, and another effort reaches the template as it
    /// came. A kwargs `enable_thinking: false` is no effort and leaves the
    /// header at the template's default.
    #[test]
    fn reasoning_effort_none_is_the_low_effort_on_glm5() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            templated(include_str!("../tests/fixtures/glm5-chat-template.jinja")),
        );
        let messages = || json!([{ "role": "user", "content": "Hi" }]);
        let header = |effort: &str| {
            format!(
                "[gMASK]<sop><|system|>Reasoning Effort: {effort}<|user|>Hi<|assistant|><think>"
            )
        };
        let (status, low) = applied(
            addr,
            &json!({ "messages": messages(), "reasoning_effort": "low" }),
        );
        assert_eq!(status, 200, "{low}");
        assert_eq!(applied_prompt(&low), header("Low"));
        for body in [
            json!({ "messages": messages(), "reasoning_effort": "none" }),
            json!({
                "messages": messages(),
                "reasoning_effort": "none",
                "chat_template_kwargs": { "reasoning_effort": "high" },
            }),
        ] {
            let (status, none) = applied(addr, &body);
            assert_eq!(status, 200, "{none}");
            // PIN(2026-10-10): user decision — a template that cannot turn thinking off gets "none" as its lowest effort, "low".
            assert_eq!(none, low, "none is the template's lowest effort: {body}");
        }
        let (status, high) = applied(
            addr,
            &json!({ "messages": messages(), "reasoning_effort": "high" }),
        );
        assert_eq!(status, 200, "{high}");
        assert_eq!(applied_prompt(&high), header("High"));
        let (status, switch) = applied(
            addr,
            &json!({ "messages": messages(), "chat_template_kwargs": { "enable_thinking": false } }),
        );
        assert_eq!(status, 200, "{switch}");
        assert_eq!(applied_prompt(&switch), header("Max"));
    }

    /// `chat_template_kwargs.enable_thinking` as a string is a 400 naming it,
    /// llama-server's refusal for a quoted bool; a bool reaches the template
    /// as today.
    #[test]
    fn a_string_enable_thinking_kwarg_is_refused_by_name() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let messages = || json!([{ "role": "user", "content": "Hi" }]);
        let (status, body) = applied(
            addr,
            &json!({
                "messages": messages(),
                "chat_template_kwargs": { "enable_thinking": "false" },
            }),
        );
        assert_eq!(status, 400, "{body}");
        assert!(
            body.contains("enable_thinking"),
            "the 400 names the field: {body}"
        );
        for on in [true, false] {
            let (status, body) = applied(
                addr,
                &json!({ "messages": messages(), "chat_template_kwargs": { "enable_thinking": on } }),
            );
            assert_eq!(status, 200, "{body}");
        }
    }

    /// With thinking off the whole output is `content`: the template closed
    /// the span in the prompt (`reasoning_effort: "none"`), so the output
    /// parser starts outside it and the think budget holds nothing. The same
    /// request without it opens the span, and the whole output is
    /// `reasoning_content`. The mock's output after the prompt's closed span
    /// is the newlines that end it, echoed, which Qwen3.8's markup drops from
    /// the start of the content as llama.cpp's parser does: no field takes them.
    #[test]
    fn thinking_off_puts_the_whole_output_in_content() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(4096)),
            templated(include_str!("../tests/fixtures/qwen38-chat-template.jinja")),
        );
        let message = |effort: &str| {
            let body = format!(
                "{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\
                 \"temperature\":0,\"max_tokens\":4,\"reasoning_budget\":2{effort}}}"
            );
            let (status, text) = roundtrip(addr, "POST", "/v1/chat/completions", &body);
            assert_eq!(status, 200, "{text}");
            serde_json::from_str::<Value>(&text).expect("JSON")["choices"][0]["message"].clone()
        };
        let on = message("");
        assert!(
            on["reasoning_content"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "thinking on: the output is reasoning: {on}"
        );
        assert_eq!(
            on["content"].as_str(),
            Some(""),
            "thinking on: no content: {on}"
        );
        let off = message(",\"reasoning_effort\":\"none\"");
        assert!(
            off.get("reasoning_content").is_none(),
            "thinking off: no reasoning: {off}"
        );
        assert_eq!(
            off["content"].as_str(),
            Some(""),
            "thinking off: the output is content, its opening newlines dropped: {off}"
        );
    }

    /// Solar-Open2's template on `engine`, served on `tags` when a seat names
    /// them ([`Server::with_think_tags`]) and on the default `<think>` tags
    /// otherwise.
    fn solar_template_server(
        engine: Box<dyn crate::Engine>,
        tags: Option<ThinkTags>,
    ) -> SocketAddr {
        let config = templated(include_str!("../tests/fixtures/solar-chat-template.jinja"));
        let mut server = Server::bind("127.0.0.1:0", engine, config).expect("bind");
        if let Some(tags) = tags {
            server = server.with_think_tags(tags);
        }
        server.spawn().expect("spawn")
    }

    /// The `message` of a chat answer to `body` on `addr`.
    fn chat_message(addr: SocketAddr, body: &str) -> Value {
        let (status, text) = roundtrip(addr, "POST", "/v1/chat/completions", body);
        assert_eq!(status, 200, "{text}");
        serde_json::from_str::<Value>(&text).expect("JSON")["choices"][0]["message"].clone()
    }

    /// The split reads the tags a seat names ([`Server::with_think_tags`]).
    /// Solar-Open2's template ends its generation prompt with `<|think:start|>`,
    /// so the output starts inside the span and `<|think:end|>` ends it. A
    /// server on the default `<think>` tags reads that prompt as one that leaves
    /// the span to the model, and the whole output is content. With
    /// `reasoning_effort: "low"` the prompt closes the span itself and nothing is
    /// split, on either tags.
    #[test]
    fn a_seat_names_the_think_tags_the_split_reads() {
        let message = |tags: Option<ThinkTags>, effort: &str| {
            let engine = crate::ScriptedEngine::new(4096, "Add.<|think:end|>Four.");
            let addr = solar_template_server(Box::new(engine), tags);
            chat_message(
                addr,
                &format!(
                    "{{\"messages\":[{{\"role\":\"user\",\"content\":\"2+2?\"}}],\
                     \"temperature\":0,\"max_tokens\":64{effort}}}"
                ),
            )
        };
        let solar = message(Some(ThinkTags::SOLAR), "");
        assert_eq!(solar["reasoning_content"], "Add.", "{solar}");
        assert_eq!(solar["content"], "Four.", "{solar}");
        let default = message(None, "");
        assert!(default.get("reasoning_content").is_none(), "{default}");
        assert_eq!(default["content"], "Add.<|think:end|>Four.", "{default}");
        let off = message(Some(ThinkTags::SOLAR), ",\"reasoning_effort\":\"none\"");
        assert!(off.get("reasoning_content").is_none(), "{off}");
        assert_eq!(off["content"], "Add.<|think:end|>Four.", "{off}");
    }

    /// A structured-output request on a prompt that left the span open has it
    /// closed first, with the seat's close: on Solar-Open2's tags the
    /// generation opens with `<|think:end|>`, so the document is the content
    /// and nothing is reasoning. A close on the default tags would stay inside
    /// the seat's span and the document would be reasoning.
    #[test]
    fn a_seats_close_precedes_a_structured_output_program() {
        let addr = solar_template_server(
            Box::new(crate::MockEngine::new(4096)),
            Some(ThinkTags::SOLAR),
        );
        let m = chat_message(
            addr,
            r#"{"messages":[{"role":"user","content":"ok?"}],"max_tokens":64,
                "reasoning_loop":true, "response_format":{"type":"json_schema",
                "json_schema":{"name":"r","schema":{"type":"boolean"}}}}"#,
        );
        let content = m["content"].as_str().unwrap_or_default().trim();
        assert!(["true", "false"].contains(&content), "{m}");
        assert_eq!(
            m["reasoning_content"].as_str().unwrap_or_default(),
            "",
            "{m}"
        );
    }

    /// The think budget runs on the seat's tags. A script that never closes
    /// its span, `reasoning_budget` 4: the first four ids are the reasoning
    /// and `<|think:end|>` is forced in place of the next 13 (the mock spells
    /// it a byte an id), so the content resumes after them. A script that
    /// closes the span itself inside a budget it does not spend: the tracker
    /// reads the seat's close, retires, and forces nothing.
    #[test]
    fn a_seats_tags_close_and_retire_the_reasoning_budget() {
        const SCRIPT: &str = "abcdefghijklmnopqrstuvwxyz0123456789";
        let budgeted = |script: &str, budget: usize| {
            let addr = solar_template_server(
                Box::new(crate::ScriptedEngine::new(4096, script)),
                Some(ThinkTags::SOLAR),
            );
            chat_message(
                addr,
                &format!(
                    "{{\"messages\":[{{\"role\":\"user\",\"content\":\"q\"}}],\
                     \"temperature\":0,\"max_tokens\":64,\"reasoning_budget\":{budget}}}"
                ),
            )
        };
        let capped = budgeted(SCRIPT, 4);
        assert_eq!(capped["reasoning_content"], "abcd", "{capped}");
        assert_eq!(
            capped["content"],
            SCRIPT[4 + ThinkTags::SOLAR.close.len()..],
            "{capped}"
        );
        let closed = format!("ab{}{}", ThinkTags::SOLAR.close, &SCRIPT[..24]);
        let natural = budgeted(&closed, 20);
        assert_eq!(natural["reasoning_content"], "ab", "{natural}");
        assert_eq!(natural["content"], SCRIPT[..24], "{natural}");
    }

    /// The loop guard (`reasoning_loop`) on every route that generates: a
    /// script that repeats `abc` in the span ends at its 259th id, the run of
    /// period 3 at 256, with the guard's reason; unguarded, or with thinking
    /// off (the prompt closed the span, so the cycle is content), it runs to
    /// its end. A guard that is not a boolean is a 400 naming it.
    #[test]
    fn the_loop_guard_ends_a_repeating_span_with_its_reason() {
        let script = "abc".repeat(200);
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::ScriptedEngine::new(4096, &script)),
            templated(include_str!("../tests/fixtures/qwen38-chat-template.jinja")),
        );
        let post = |path: &str, mut body: Value, guard: Value| {
            body["reasoning_loop"] = guard;
            let (status, text) = roundtrip(addr, "POST", path, &body.to_string());
            (status, serde_json::from_str::<Value>(&text).expect("JSON"))
        };
        let chat = |guard: bool, effort: &str| {
            let body = json!({"messages": [{"role": "user", "content": "q"}],
                "max_tokens": 1024, "reasoning_effort": effort});
            let (status, v) = post("/v1/chat/completions", body, json!(guard));
            assert_eq!(status, 200, "{v}");
            v["choices"][0].clone()
        };
        let looped = chat(true, "high");
        assert_eq!(looped["finish_reason"], "reasoning_loop", "{looped}");
        assert_eq!(looped["message"]["reasoning_content"], script[..259]);
        let free = chat(false, "high");
        assert_eq!(free["finish_reason"], "stop", "{free}");
        assert_eq!(free["message"]["reasoning_content"], script);
        let off = chat(true, "none");
        assert_eq!(off["finish_reason"], "stop", "{off}");
        assert_eq!(off["message"]["content"], script);
        let (_, r) = post(
            "/v1/responses",
            json!({"input": "q", "max_output_tokens": 1024}),
            json!(true),
        );
        assert_eq!(r["status"], "incomplete", "{r}");
        assert_eq!(r["incomplete_details"]["reason"], "reasoning_loop", "{r}");
        let (_, m) = post(
            "/v1/messages",
            json!({"messages": [{"role": "user", "content": "q"}], "max_tokens": 1024}),
            json!(true),
        );
        assert_eq!(m["stop_reason"], "reasoning_loop", "{m}");
        let (_, c) = post(
            "/completion",
            json!({"prompt": "q<think>", "n_predict": 1024}),
            json!(true),
        );
        assert_eq!(c["stop_type"], "reasoning_loop", "{c}");
        assert_eq!(c["tokens_predicted"], 259, "{c}");
        let (status, e) = post(
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "q"}]}),
            json!(1),
        );
        assert_eq!(status, 400, "{e}");
        assert_eq!(e["error"]["param"], "reasoning_loop", "{e}");
    }

    /// The media part's refusals are the request's 400 with the error's own text, about the
    /// field that carried the media. An empty span is the model's 500, about no field. A
    /// numbered placeholder whose ids are not its image's position is the request's 400, with
    /// the other pairing faults.
    #[test]
    fn an_empty_span_is_500_and_a_wrong_image_number_is_400() {
        let empty = media_error(&MediaError::Expand(ExpandError::EmptySpan(0)), "messages");
        assert_eq!(
            (empty.code, empty.kind, empty.message.as_str()),
            (500, "server_error", "image 0 has an empty span")
        );
        assert_eq!(empty.param, None);
        let number = media_error(&MediaError::ImageNumber { index: 2 }, "messages");
        assert_eq!((number.code, number.kind), (400, "invalid_request_error"));
        assert_eq!(number.param.as_deref(), Some("messages"));
        assert_eq!(
            number.message,
            MediaError::ImageNumber { index: 2 }.to_string()
        );
        for e in [
            MediaError::Expand(ExpandError::PlaceholderCount {
                placeholders: 1,
                images: 0,
            }),
            MediaError::ImageLast,
        ] {
            let got = media_error(&e, "messages");
            assert_eq!((got.code, got.kind), (400, "invalid_request_error"));
            assert_eq!(got.message, e.to_string());
        }
    }

    /// The mock engine behind a media model (a model that numbers its images,
    /// [`crate::media::NumberedMedia`], or one that takes clips,
    /// [`crate::media::ClipMedia`]); only the render is asked of it.
    struct MediaEngine(crate::MockEngine, crate::media::SharedMediaModel);

    impl Engine for MediaEngine {
        fn tokenizer(&self) -> Arc<dyn crate::engine::Tokenizer> {
            self.0.tokenizer()
        }

        fn media_model(&self) -> Option<crate::media::SharedMediaModel> {
            Some(Arc::clone(&self.1))
        }

        fn prefill(&mut self, ids: &[u32]) -> Result<(), crate::engine::EngineError> {
            self.0.prefill(ids)
        }

        fn next(
            &mut self,
            last: u32,
            logits_out: Option<&mut [f32]>,
        ) -> Result<u32, crate::engine::EngineError> {
            self.0.next(last, logits_out)
        }

        fn reset(&mut self) -> Result<(), crate::engine::EngineError> {
            self.0.reset()
        }

        fn ctx_max(&self) -> usize {
            self.0.ctx_max()
        }

        fn describe(&self) -> String {
            self.0.describe()
        }
    }

    /// One counter runs through the whole request: a model that numbers its images gets image 0 in
    /// the first user message and image 1 in the second, and the images are handed on in that
    /// order [FAIL-first: a count that restarts at each message renders both as 0].
    #[test]
    fn images_are_numbered_across_the_messages_of_a_request() {
        let (_addr, state, _ended) = super::testserve::spawn(
            Box::new(MediaEngine(
                crate::MockEngine::new(64),
                Arc::new(crate::media::NumberedMedia),
            )),
            super::testserve::mock_config(),
        );
        let image = |url: &str| json!({"type": "image_url", "image_url": {"url": url}});
        let body = json!({"messages": [
            {"role": "user", "content": [image("data:a")]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "and"}, image("data:b")]},
        ]});
        let Value::Object(body) = body else {
            unreachable!("an object literal")
        };
        let rendered =
            super::render_chat(&state, &body).unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(
            rendered.text,
            "<user>\n<image_id>0</image_id><image><unk>\
             <assistant>\nok\
             <user>\nand\n<image_id>1</image_id><image><unk>\
             <assistant>\n"
        );
        let urls: Vec<&str> = rendered
            .media
            .iter()
            .filter_map(|m| match m {
                super::Medium::Image(url) => Some(url.as_str()),
                super::Medium::Video(_) => None,
            })
            .collect();
        assert_eq!(urls, ["data:a", "data:b"]);
    }

    /// A server over `engine` whose video is `video`: the address, state and end channel.
    fn spawn_video(
        engine: Box<dyn Engine>,
        video: crate::video::VideoSupport,
    ) -> (
        SocketAddr,
        Arc<super::State>,
        std::sync::mpsc::Receiver<End>,
    ) {
        let server = super::Server::bind("127.0.0.1:0", engine, super::testserve::mock_config())
            .expect("bind")
            .with_video(video);
        let addr = server.local_addr().expect("addr");
        let (state, ended) = server.start().expect("start");
        (addr, state, ended)
    }

    fn clip_engine(ctx: usize) -> Box<dyn Engine> {
        Box::new(MediaEngine(
            crate::MockEngine::new(ctx),
            Arc::new(crate::media::ClipMedia),
        ))
    }

    /// `/props` names `modalities.video` only when a decoder and a video rule are both there: a
    /// server whose ffmpeg is absent says false whatever its model takes, and so does a model
    /// with no rule or none at all, while `vision` keeps its own meaning.
    #[test]
    fn props_names_video_only_when_a_decoder_and_a_rule_are_both_there() {
        use crate::video::{MockClips, VideoSupport};
        let mock = MockClips::new();
        let numbered: Box<dyn Engine> = Box::new(MediaEngine(
            crate::MockEngine::new(64),
            Arc::new(crate::media::NumberedMedia),
        ));
        let cases: [(Box<dyn Engine>, VideoSupport, [bool; 2]); 4] = [
            (clip_engine(64), mock.support(1), [true, true]),
            (
                clip_engine(64),
                VideoSupport::absent("not found"),
                [true, false],
            ),
            (numbered, mock.support(1), [true, false]),
            (
                Box::new(crate::MockEngine::new(64)),
                mock.support(1),
                [false, false],
            ),
        ];
        for (engine, video, [vision, takes]) in cases {
            let (addr, _state, _ended) = spawn_video(engine, video);
            let (status, body) = super::testserve::roundtrip(addr, "GET", "/props", &[], "");
            assert_eq!(status, 200, "{body}");
            let props: Value = serde_json::from_str(&body).expect("JSON");
            assert_eq!(
                props["modalities"],
                json!({"vision": vision, "audio": false, "video": takes}),
                "{body}"
            );
        }
    }

    /// A chat's clip is probed and planned under the context its request's text leaves, before
    /// the template renders: the group count follows the budget, the text holds a placeholder per
    /// group, and `/apply-template` returns that text without decoding a frame.
    #[test]
    fn a_chat_clip_is_planned_under_the_context_its_text_leaves() {
        use crate::video::{MockClips, PROMPT_FRAME_TOKENS};
        let mock = MockClips::new();
        let said = crate::MockEngine::new(8).tokenizer().encode("hi").len();
        assert!(
            said >= 2,
            "the text must cost context for the budget to show it"
        );
        let clip = json!({"type": "input_video", "input_video": {
            "data": crate::media::encode_base64(b"30 10/1 4x2"), "format": "mp4"
        }});
        let body = |ctx_text: &str| {
            let Value::Object(b) = json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": ctx_text}, clip.clone()
            ]}]}) else {
                unreachable!("an object literal")
            };
            b
        };
        // Room for the whole clip (three groups of two tokens), then for four tokens, then none.
        for (room, markers) in [(40, 3), (4, 2)] {
            let (_addr, state, _ended) = spawn_video(
                clip_engine(PROMPT_FRAME_TOKENS + said + room),
                mock.support(2),
            );
            let rendered =
                super::render_chat(&state, &body("hi")).unwrap_or_else(|e| panic!("{}", e.message));
            assert_eq!(rendered.text.matches("<vid>").count(), markers, "{room}");
            assert!(
                rendered.text.starts_with("<user>\nhi\n["),
                "{}",
                rendered.text
            );
        }
        let (_addr, state, _ended) =
            spawn_video(clip_engine(PROMPT_FRAME_TOKENS + said + 1), mock.support(2));
        let refused = super::render_chat(&state, &body("hi"))
            .err()
            .expect("no room");
        assert_eq!(refused.code, 400);
        assert!(
            refused.message.starts_with("video: "),
            "{}",
            refused.message
        );
        // `/apply-template` renders the placeholders and decodes nothing.
        let (addr, _state, _ended) = spawn_video(
            clip_engine(PROMPT_FRAME_TOKENS + said + 40),
            mock.support(2),
        );
        let request = Value::Object(body("hi")).to_string();
        let (status, text) =
            super::testserve::roundtrip(addr, "POST", "/apply-template", &[], &request);
        assert_eq!(status, 200, "{text}");
        let prompt: Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(
            prompt["prompt"],
            "<user>\nhi\n[<vid>,<vid>,<vid>]<assistant>\n"
        );
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    /// Where ffmpeg is absent a chat clip is the request's 400 with the named remedy, whatever
    /// the model takes.
    #[test]
    fn a_chat_clip_without_ffmpeg_is_a_named_400() {
        let (_addr, state, _ended) = spawn_video(
            clip_engine(400),
            crate::video::VideoSupport::absent("ffmpeg: not found"),
        );
        let Value::Object(body) = json!({"messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": "data:video/mp4;base64,AAAA"}
        ]}]}) else {
            unreachable!("an object literal")
        };
        let e = super::render_chat(&state, &body).err().expect("no decoder");
        assert_eq!(e.code, 400);
        assert_eq!(
            e.message,
            "video input needs ffmpeg and ffprobe on PATH (ffmpeg: not found); install ffmpeg, \
             or send the clip as decoded one-FPS image_url frames after a Video: text part"
        );
    }

    /// One stderr line a finished request, whatever endpoint it came
    /// through: the chat route and `/v1/messages` each print exactly one,
    /// naming the endpoint's own path.
    #[test]
    fn one_stderr_line_a_finished_request() {
        let (addr, state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let chat = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/v1/chat/completions", &[], chat);
        assert_eq!(status, 200, "{body}");
        let messages = r#"{"max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#;
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/v1/messages", &[], messages);
        assert_eq!(status, 200, "{body}");
        let lines = state.shared.notes();
        let finish: Vec<&String> = lines.iter().filter(|l| l.contains("; stop ")).collect();
        assert_eq!(finish.len(), 2, "one line a request: {lines:#?}");
        assert!(
            finish[0].starts_with("bloomery-serve: request ")
                && finish[0].contains(" /v1/chat/completions: prompt "),
            "{}",
            finish[0]
        );
        assert!(
            finish[1].contains(" /v1/messages: prompt "),
            "{}",
            finish[1]
        );
    }

    /// The choices of one request are tasks of their own: each finish line
    /// names its choice's index after a task id no other choice shares, the
    /// id `/slots` and the live view tell one generation by.
    #[test]
    fn each_choice_has_its_own_task_id() {
        let (addr, state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let ask = r#"{"prompt":"I believe the meaning of life is","max_tokens":3,"n":2}"#;
        let (status, body) = super::testserve::roundtrip(addr, "POST", "/v1/completions", &[], ask);
        assert_eq!(status, 200, "{body}");
        let lines = state.shared.notes();
        let mut tags: Vec<(String, String)> = lines
            .iter()
            .filter(|l| l.contains("; stop "))
            .map(|l| {
                let tag = l
                    .strip_prefix("bloomery-serve: request ")
                    .and_then(|r| r.split_whitespace().next())
                    .unwrap_or_else(|| panic!("no request tag: {l}"));
                let (id, choice) = tag
                    .split_once('.')
                    .unwrap_or_else(|| panic!("no choice index: {l}"));
                (choice.to_owned(), id.to_owned())
            })
            .collect();
        tags.sort();
        assert_eq!(tags.len(), 2, "one line a choice: {lines:#?}");
        assert_eq!(
            (tags[0].0.as_str(), tags[1].0.as_str()),
            ("0", "1"),
            "{lines:#?}"
        );
        assert_ne!(
            tags[0].1, tags[1].1,
            "each choice its own task id: {lines:#?}"
        );
    }

    /// The finish line's counts are the ones `/metrics` books for the
    /// request: its prompt and generated ids equal the counters' move over
    /// the request.
    #[test]
    fn the_stderr_line_counts_what_metrics_books() {
        let (addr, state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let before = super::testserve::roundtrip(addr, "GET", "/metrics", &[], "").1;
        let ask = r#"{"prompt":"I believe the meaning of life is","max_tokens":4}"#;
        let (status, body) = super::testserve::roundtrip(addr, "POST", "/completion", &[], ask);
        assert_eq!(status, 200, "{body}");
        let after = super::testserve::roundtrip(addr, "GET", "/metrics", &[], "").1;
        let counter = |text: &str, name: &str| -> u64 {
            let row = format!("llamacpp:{name} ");
            text.lines()
                .find_map(|l| l.strip_prefix(row.as_str()))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or_else(|| panic!("no {name} row in the metrics: {text}"))
        };
        let lines = state.shared.notes();
        let finish: Vec<&String> = lines.iter().filter(|l| l.contains("; stop ")).collect();
        assert_eq!(finish.len(), 1, "one line the request: {lines:#?}");
        let line = finish[0];
        let count = |word: &str| -> u64 {
            line.split_whitespace()
                .skip_while(|t| !t.starts_with(word))
                .nth(1)
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| panic!("no {word} count in the line: {line}"))
        };
        assert_eq!(
            count("prompt"),
            counter(&after, "prompt_tokens_total") - counter(&before, "prompt_tokens_total"),
            "the line's prompt count is what /metrics added: {line}"
        );
        assert_eq!(
            count("generated"),
            counter(&after, "tokens_predicted_total") - counter(&before, "tokens_predicted_total"),
            "the line's generated count is what /metrics added: {line}"
        );
    }

    /// `/metrics` splits the decode rounds by width and the engine's waits
    /// by phase, and the draft's counts by depth [FAIL-first: no labelled
    /// row]: two sampled choices on two slots pass in rounds of two, a
    /// round's card wait is the engine's count of the passes in it, the
    /// prompts' PLE waits stay out of the rounds, the rounds and the prompt
    /// calls are every call and slot `n_decode_total` and
    /// `n_busy_slots_per_decode` count, and the counts by depth, under the
    /// drafter's kind, sum to the draft's totals. A slot action's wait shows
    /// in the next scrape with no generation between.
    #[test]
    fn metrics_split_rounds_by_width_waits_by_phase_and_drafts_by_depth() {
        use crate::engine::{Drafted, Sampler, Tokenizer, Waits};
        const CARD: Duration = Duration::from_millis(1);
        const PLE: Duration = Duration::from_millis(3);
        /// A drafting mock that counts [`CARD`] of card wait a generating
        /// call or a reset and [`PLE`] of PLE wait a prompt call.
        struct Waiting {
            inner: crate::DraftMock,
            waits: Waits,
        }
        impl Engine for Waiting {
            fn tokenizer(&self) -> Arc<dyn Tokenizer> {
                self.inner.tokenizer()
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<(), super::EngineError> {
                self.waits.ple += PLE;
                self.inner.prefill(ids)
            }
            fn next(
                &mut self,
                last: u32,
                out: Option<&mut [f32]>,
            ) -> Result<u32, super::EngineError> {
                self.waits.card += CARD;
                self.inner.next(last, out)
            }
            fn advance(
                &mut self,
                last: u32,
                out: &mut Vec<u32>,
            ) -> Result<Drafted, super::EngineError> {
                self.waits.card += CARD;
                self.inner.advance(last, out)
            }
            fn advance_sampled(
                &mut self,
                last: u32,
                history: Vec<u32>,
                sampler: Sampler,
                out: &mut Vec<u32>,
            ) -> (Result<Drafted, super::EngineError>, Vec<u32>, Sampler) {
                self.waits.card += CARD;
                self.inner.advance_sampled(last, history, sampler, out)
            }
            fn advance_rows(&self) -> usize {
                self.inner.advance_rows()
            }
            fn drafts_sampled(&self) -> bool {
                self.inner.drafts_sampled()
            }
            fn slots(&self) -> usize {
                self.inner.slots()
            }
            fn select_slot(&mut self, slot: usize) -> Result<(), super::EngineError> {
                self.inner.select_slot(slot)
            }
            fn slot_drafts(&self) -> bool {
                self.inner.slot_drafts()
            }
            fn reset(&mut self) -> Result<(), super::EngineError> {
                self.waits.card += CARD;
                self.inner.reset()
            }
            fn ctx_max(&self) -> usize {
                self.inner.ctx_max()
            }
            fn describe(&self) -> String {
                self.inner.describe()
            }
            fn props_engine(&self) -> EngineProps {
                let mut props = self.inner.props_engine();
                props.draft.as_mut().expect("drafting mock").kind =
                    Some("draft\"v2\\path\nnext\\n".into());
                props
            }
            fn waits(&self) -> Waits {
                self.waits
            }
        }
        let engine = Waiting {
            inner: crate::DraftMock::new(256)
                .with_slots(2)
                .with_sampled_draft(3, 2, 3),
            waits: Waits::default(),
        };
        let mut config = super::testserve::mock_config();
        config.sampler = Some(crate::sampling::reference_factory());
        // Slot actions are served with a save folder; the erase writes none.
        config.slot_save_path = Some(std::env::temp_dir());
        let slots = SlotConfig {
            parallel: 2,
            ..SlotConfig::default()
        };
        let server =
            Server::bind_with("127.0.0.1:0", Box::new(engine), config, slots).expect("bind");
        let addr = server.local_addr().expect("addr");
        let (_state, _ended) = server.start().expect("start");
        let ask = r#"{"prompt":"abcabcabc","max_tokens":12,"n":2,"temperature":0.8,"seed":5}"#;
        let (status, body) = super::testserve::roundtrip(addr, "POST", "/v1/completions", &[], ask);
        assert_eq!(status, 200, "{body}");
        let scrape = || super::testserve::roundtrip(addr, "GET", "/metrics", &[], "").1;
        let m = scrape();
        let rows = |name: &str| -> Vec<(String, f64)> {
            let head = format!("llamacpp:{name}{{");
            m.lines()
                .filter_map(|l| l.strip_prefix(head.as_str()))
                .map(|l| {
                    let (labels, value) = l.split_once("} ").expect("a labelled row");
                    (labels.to_owned(), value.parse().expect("a number"))
                })
                .collect()
        };
        let value = |m: &str, name: &str| -> f64 {
            let head = format!("llamacpp:{name} ");
            m.lines()
                .find_map(|l| l.strip_prefix(head.as_str()))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("no {name} row: {m}"))
        };
        let plain = |name: &str| value(&m, name);
        let width = |labels: &str| -> f64 {
            labels
                .strip_prefix("width=\"")
                .and_then(|w| w.strip_suffix('"'))
                .and_then(|w| w.parse().ok())
                .unwrap_or_else(|| panic!("no width in {labels}"))
        };
        let rounds = rows("rounds_total");
        assert!(
            rounds.iter().any(|(l, n)| width(l) == 2.0 && *n > 0.0),
            "the two choices passed together: {m}"
        );
        let prompts = 2.0;
        let calls: f64 = rounds.iter().map(|(_, n)| n).sum();
        let carried: f64 = rounds.iter().map(|(l, n)| width(l) * n).sum();
        assert_eq!(calls + prompts, plain("n_decode_total"), "{m}");
        assert!(
            ((carried + prompts) / (calls + prompts) - plain("n_busy_slots_per_decode")).abs()
                < 1e-9,
            "{m}"
        );
        let card = CARD.as_secs_f64();
        let round_card = rows("round_card_wait_seconds_total");
        assert_eq!(
            round_card.len(),
            rounds.len(),
            "one card-wait row a width: {m}"
        );
        for ((l, n), (cl, waited)) in rounds.iter().zip(&round_card) {
            assert_eq!(l, cl, "{m}");
            assert!(
                (waited - width(l) * n * card).abs() < 1e-9,
                "a round of {l} waited a card call a slot: {m}"
            );
        }
        assert_eq!(
            rows("round_seconds_total").len(),
            rounds.len(),
            "one round-time row per width: {m}"
        );
        assert_eq!(
            rows("round_ple_wait_seconds_total").len(),
            rounds.len(),
            "one PLE-wait row per width: {m}"
        );
        assert!(
            rows("round_ple_wait_seconds_total")
                .iter()
                .all(|(_, w)| *w == 0.0),
            "no prompt ran inside a round: {m}"
        );
        assert!(
            (plain("ple_wait_seconds_total") - prompts * PLE.as_secs_f64()).abs() < 1e-9,
            "a prompt call's PLE wait: {m}"
        );
        let in_rounds: f64 = round_card.iter().map(|(_, w)| w).sum();
        assert!(plain("card_wait_seconds_total") >= in_rounds, "{m}");
        let proposed = rows("spec_decode_num_draft_tokens_by_depth_total");
        let accepted = rows("spec_decode_num_accepted_tokens_by_depth_total");
        let at = |d: usize| format!(r#"source="draft\"v2\\path\nnext\\n",depth="{d}""#);
        assert_eq!(
            proposed.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(),
            (1..=3).map(at).collect::<Vec<_>>(),
            "three depths under the drafter's kind: {m}"
        );
        assert_eq!(accepted.len(), proposed.len(), "{m}");
        for (d, ((label, p), (accepted_label, a))) in proposed.iter().zip(&accepted).enumerate() {
            assert_eq!(label, accepted_label, "{m}");
            assert!(a <= p, "depth {}: {m}", d + 1);
        }
        let sum = |rows: &[(String, f64)]| -> f64 { rows.iter().map(|(_, n)| n).sum() };
        assert_eq!(
            sum(&proposed),
            plain("spec_decode_num_draft_tokens_total"),
            "{m}"
        );
        assert_eq!(
            sum(&accepted),
            plain("spec_decode_num_accepted_tokens_total"),
            "{m}"
        );
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/slots/0?action=erase", &[], "");
        assert_eq!(status, 200, "{body}");
        let erased = scrape();
        assert!(
            (value(&erased, "card_wait_seconds_total") - plain("card_wait_seconds_total") - card)
                .abs()
                < 1e-9,
            "the erase's reset waited a card call: {erased}"
        );
    }

    /// A request the token limit ends says `length`: the line's stop cause
    /// is the generation's own.
    #[test]
    fn a_length_limited_request_says_length_in_its_stderr_line() {
        let (addr, state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            super::testserve::mock_config(),
        );
        let chat = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
        let (status, body) =
            super::testserve::roundtrip(addr, "POST", "/v1/chat/completions", &[], chat);
        assert_eq!(status, 200, "{body}");
        let lines = state.shared.notes();
        let finish = lines
            .iter()
            .find(|l| l.contains("; stop "))
            .expect("the finish line");
        assert!(
            finish.contains("stop length;"),
            "the limit names itself: {finish}"
        );
    }
}
