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
//! before it reaches the engine, and generation stops at `ctx_max` with
//! `truncated`, whatever `max_tokens` asked. Every slot has the whole
//! `ctx_max`; it is not split among them.
//!
//! At most [`MAX_CONNECTIONS`] connections are served at once, one thread each;
//! a connection past them gets a 503 with `Retry-After` and is closed. A
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

use std::fmt;
use std::io::{self, BufRead, BufReader, Read};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jinja::{ChatTemplate, TemplateError};
use serde_json::{Map, Value, json};

use crate::dsml::{ChatParser, MarkupError, Message, ToolCall, ToolFormat, Tools};
use crate::engine::{
    DraftProps, Engine, EngineError, EngineProps, ModelProps, PlacementProps, SamplerFactory,
    SamplingParams, StateError, Tokenizer,
};
use crate::flag::ApiKeys;
use crate::genloop::{self, Event, GenError, GenParams, Outcome, Slot, Timings, ms_since};
use crate::glmxml::ArgTypes;
use crate::http::{self, EventStream, Request};
use crate::media::{self, MediaError, Prompt, SharedMediaModel};
use crate::qwenxml::ParamKinds;
use crate::reasoning::{ReasoningFormat, ThinkEntry};
use crate::sampling;
use crate::sched::{Board, Refusal, Reserve, SlotConfig, Use, default_depth};
use crate::slotfile;
use crate::swap::Park;
use crate::worker::{self, Acted, Action, Msg, Shared, Submit};

/// Anthropic's Messages API, a child of this module so it runs the chat path's
/// own steps.
#[path = "anthropic.rs"]
mod anthropic;
/// Per-token log-probabilities, which the generation loop collects and the
/// completion and chat answers render.
#[path = "logprobs.rs"]
pub(crate) mod logprobs;
/// OpenAI's text completion API and the chat token counts, a child of this
/// module so they run the completion and chat paths' own steps.
#[path = "oaicompl.rs"]
mod oaicompl;
/// OpenAI's Responses API, a child of this module so it runs the chat path's
/// own steps.
#[path = "responses.rs"]
mod responses;
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

/// How long a refused connection is written to and read from: the read consumes
/// the request it sent, so the close does not reset the connection before the
/// client reads the 503.
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
    listener: TcpListener,
    state: Arc<State>,
    ended: mpsc::Receiver<End>,
    /// The engine and its slots, until [`Server::run`] starts their thread.
    slot: Slot,
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
    /// can reach are refused by name ([`ServeError::Slots`]).
    pub fn bind_with(
        addr: impl ToSocketAddrs,
        engine: Box<dyn Engine>,
        config: ServerConfig,
        slots: SlotConfig,
    ) -> Result<Self, ServeError> {
        let template = ChatTemplate::parse(&config.chat_template)?;
        if let Some(dir) = config.slot_save_path.as_ref().filter(|d| !d.is_dir()) {
            return Err(ServeError::SlotSavePath(dir.clone()));
        }
        let depth = check_slots(&*engine, &slots)?;
        let listener = TcpListener::bind(addr)?;
        let tok = engine.tokenizer();
        if tok.stops().is_empty() {
            return Err(ServeError::NoStops);
        }
        let engine_props = engine_object(&engine.props_engine());
        let media = engine.media_model();
        let info = ModelInfo {
            n_vocab: tok.n_vocab(),
            ctx_max: engine.ctx_max(),
            bos_text: tok.decode(&[tok.bos()]),
            eos_text: tok.decode(&[tok.eos()]),
        };
        let (end_tx, ended) = mpsc::channel();
        let parallel = slots.parallel;
        let shared = Shared::new(
            Board::new(parallel, depth, slots.picker),
            end_tx,
            config.sampler.unwrap_or_else(sampling::reference_factory),
        );
        let state = State {
            shared: Arc::new(shared),
            parallel,
            tok,
            fatal_linger: config.fatal_linger,
            tool_format: ToolFormat::of_chat_template(&template),
            template,
            alias: config.model_alias,
            model_path: config.model_path,
            engine_props,
            media,
            slot_save_path: config.slot_save_path,
            api_keys: config.api_keys,
            info,
            start_unix: unix_now(),
        };
        Ok(Server {
            listener,
            state: Arc::new(state),
            ended,
            slot: Slot::with_slots(engine, parallel),
        })
    }

    /// The bound address (the real port when bound to port 0).
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Starts the engine thread and the accept loop, which serves connections,
    /// one thread each and at most [`MAX_CONNECTIONS`] at once, until the
    /// listener fails; returns the server's state and what ends it. The
    /// engine thread owns the engine from here and runs detached (its handle
    /// dropped): an orderly stop waits for the loop it leaves
    /// ([`Shared::wait_loop_end`]), never for the thread's end.
    fn start(self) -> io::Result<(Arc<State>, mpsc::Receiver<End>)> {
        let Server {
            listener,
            state,
            ended,
            slot,
        } = self;
        let shared = Arc::clone(&state.shared);
        thread::Builder::new()
            .name("serve-engine".to_owned())
            .spawn(move || worker::serve(slot, shared))
            .map_err(|e| {
                io::Error::new(e.kind(), format!("cannot start the engine thread: {e}"))
            })?;
        let conn_state = Arc::clone(&state);
        accept(listener, state.shared.end.clone(), move |stream| {
            keep_alive(stream, |req, w| route(&conn_state, req, w));
        });
        Ok((state, ended))
    }

    /// Serves until the listener fails, the engine does, or the server is
    /// asked to stop, and returns why. An orderly stop never returns: it ends
    /// the process inside [`exit_shutdown`], so every seat's
    /// `Ok(server.run())` is the error paths' alone.
    pub fn run(self) -> ServeError {
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
        let (state, ended) = match self.start() {
            Ok(s) => s,
            Err(e) => return ServeError::Io(e),
        };
        // `state` holds a sender, so the channel cannot close while we wait.
        match wait_end(&ended, state.fatal_linger) {
            Ended::Error(e) => e,
            Ended::Stop(cause) => exit_shutdown(&cause, Some(&state.shared)),
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
    /// The engine's vocabulary, read without the engine.
    tok: Arc<dyn Tokenizer>,
    fatal_linger: Duration,
    template: ChatTemplate,
    /// The tool-call markup the template teaches, read from its source once.
    tool_format: ToolFormat,
    alias: String,
    model_path: String,
    /// `/props`' `engine` object, built when the server binds.
    engine_props: Value,
    /// The model's image input, `None` for an engine that takes none: an
    /// image part is rendered as the model's placeholder only when it exists,
    /// and refused by name otherwise. Read without the engine.
    media: Option<SharedMediaModel>,
    /// Where slot files live; `None` refuses slot actions.
    slot_save_path: Option<PathBuf>,
    /// The API keys every request is checked against before it dispatches;
    /// an empty set checks nothing.
    api_keys: ApiKeys,
    info: ModelInfo,
    start_unix: u64,
}

pub(crate) fn relock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
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

/// One live connection's place under [`MAX_CONNECTIONS`], given back when its
/// thread ends, by a return or a panic.
struct Permit(Arc<AtomicUsize>);

impl Permit {
    /// A place, or `None` when [`MAX_CONNECTIONS`] are live.
    fn take(live: &Arc<AtomicUsize>) -> Option<Permit> {
        let held = live.fetch_add(1, Ordering::SeqCst);
        let permit = Permit(Arc::clone(live));
        (held < MAX_CONNECTIONS).then_some(permit)
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
    let serve = Arc::new(serve);
    thread::spawn(move || {
        let mut repeats = 0;
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    repeats = 0;
                    admit(&serve, &live, port, stream);
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

// ---------------------------------------------------------------- orderly stop

/// How long an orderly stop waits for the engine thread to leave its loop
/// ([`Shared::wait_loop_end`]). The worker checks the stop between engine
/// calls, so the wait covers the one call it can be inside of when the stop
/// arrives — a whole-prompt call at a slot's full context, the longest single
/// call the seats run — plus the booking of the requests it ends; every
/// seat's decode step sits far under it. The engine's Drop after the loop is
/// not waited for: freeing a host tier's pinned pages and joining its
/// threads can outlast the bound, and the process exit reclaims the engine's
/// memory anyway.
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
    arm(libc::SIGINT);
    arm(libc::SIGTERM);
}

/// Registers `on_signal` for `sig` with `SA_RESTART`, so a signal arriving
/// mid-`accept` or mid-`read` restarts the call instead of failing it: the
/// stop is carried by the pipe, not by an EINTR.
fn arm(sig: libc::c_int) {
    // SAFETY: an all-zero `sigaction` is a valid value — a null handler, an
    // empty mask, no flags — and only the fields set below are changed.
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    act.sa_sigaction = on_signal as *const () as usize;
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

/// The orderly stop's last act, the one owner of the process's exit: one
/// stderr line naming the cause — and whether the engine thread left its
/// loop, when there is one to wait for — then success. Called from `run`,
/// after the stop's initiator already refused new work and told the engine
/// thread to end, so no seat's `Ok(server.run())` return is reached by it.
/// The engine's Drop is left to run beside the exit: the wait ends at the
/// loop ([`Shared::wait_loop_end`]), not at the thread's end.
pub(crate) fn exit_shutdown(cause: &Stop, shared: Option<&Shared>) -> ! {
    let ended = shared.map(|sh| sh.wait_loop_end(ENGINE_STOP));
    let tail = match ended {
        Some(false) => {
            format!(
                "; the engine thread did not end within {} s",
                ENGINE_STOP.as_secs()
            )
        }
        _ => String::new(),
    };
    eprintln!("bloomery-serve: shutdown ({cause}){tail}");
    std::process::exit(0)
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

/// Serves `stream` on a thread of its own, named `serve:<port>`, or refuses it
/// with a 503 when [`MAX_CONNECTIONS`] are live or no thread can be started.
fn admit<F>(serve: &Arc<F>, live: &Arc<AtomicUsize>, port: u16, stream: TcpStream)
where
    F: Fn(TcpStream) + Send + Sync + 'static,
{
    let Some(permit) = Permit::take(live) else {
        refuse(
            stream,
            &format!("the server is serving {MAX_CONNECTIONS} connections, its limit"),
        );
        return;
    };
    // The spawn consumes the stream even when it fails; the clone answers then.
    let answer = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            refuse(stream, &format!("cannot hold the connection: {e}"));
            return;
        }
    };
    let serve = Arc::clone(serve);
    let spawned = thread::Builder::new()
        .name(format!("serve:{port}"))
        .spawn(move || {
            let _permit = permit;
            serve(stream);
        });
    if let Err(e) = spawned {
        refuse(answer, &format!("cannot start a connection thread: {e}"));
    }
}

/// Answers a connection the server does not serve with a 503 carrying
/// `Retry-After`, then closes it after reading what the client sent, for at most
/// about [`REFUSE_DRAIN`] twice.
fn refuse(mut stream: TcpStream, message: &str) {
    let _ = stream.set_write_timeout(Some(REFUSE_DRAIN));
    let body = error_body(503, "unavailable_error", message);
    let sent = http::respond(
        &mut stream,
        &closing(),
        503,
        JSON,
        &[("Retry-After", RETRY_AFTER_SECS.to_owned())],
        body.to_string().as_bytes(),
    );
    if sent.is_err() || stream.shutdown(Shutdown::Write).is_err() {
        return;
    }
    if stream.set_read_timeout(Some(REFUSE_DRAIN)).is_err() {
        return;
    }
    let until = Instant::now() + REFUSE_DRAIN;
    let mut sink = [0u8; 4096];
    while Instant::now() < until {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
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

/// Answers `stream`'s requests in order through `route` while the client keeps
/// the connection alive ([`next_request_arrives`]); `route` answers one
/// request, `Ok(true)` when the connection may carry another. A request that
/// cannot be read is a 400 and the connection closes.
pub(crate) fn keep_alive(
    stream: TcpStream,
    mut route: impl FnMut(&Request, &mut TcpStream) -> io::Result<bool>,
) {
    let _ = stream.set_nodelay(true);
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::new(stream);
    loop {
        if !next_request_arrives(&mut r) {
            return;
        }
        let req = match http::read_request(&mut r, &mut w) {
            Ok(Some(req)) => req,
            Ok(None) => return,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                let body = error_body(400, "invalid_request_error", &e.to_string());
                let _ = http::respond(
                    &mut w,
                    &closing(),
                    400,
                    JSON,
                    &[],
                    body.to_string().as_bytes(),
                );
                return;
            }
            Err(_) => return,
        };
        match route(&req, &mut w) {
            Ok(true) if req.keep_alive => {}
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
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError {
        retry_after: false,
        code: 400,
        kind: "invalid_request_error",
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
    let body = error_body(e.code, e.kind, &e.message).to_string();
    let retry = [("Retry-After", RETRY_AFTER_SECS.to_owned())];
    let headers: &[(&str, String)] = if e.retry_after { &retry } else { &[] };
    http::respond(w, req, e.code, JSON, headers, body.as_bytes())?;
    Ok(true)
}

/// The headers of the 204 an `OPTIONS` request gets: any origin may send
/// `GET`, `POST` and `OPTIONS` with any header.
pub(crate) fn cors_preflight() -> [(&'static str, String); 2] {
    [
        (
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS".to_owned(),
        ),
        ("Access-Control-Allow-Headers", "*".to_owned()),
    ]
}

/// The one stderr line a request the key check refuses writes: its method,
/// path and peer, and never the key it sent.
fn deny_line(method: &str, path: &str, peer: &str) -> String {
    format!("{method} {path} from {peer}: refused, no valid API key")
}

/// The key check before either server's dispatch — this server's
/// [`route`] and the decision server's own connection handler, the one call
/// both make. A request that passes ([`ApiKeys::allows`]) is `Ok(false)`;
/// one that does not is answered with llama-server's 401 (`Invalid API Key`,
/// `authentication_error`) after the one [`deny_line`], and is `Ok(true)`:
/// answered, and the connection may carry another request.
pub(crate) fn key_denied(keys: &ApiKeys, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    if keys.allows(
        &req.method,
        &req.path,
        req.header("Authorization"),
        req.header("X-Api-Key"),
    ) {
        return Ok(false);
    }
    let peer = w
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|e| format!("a peer that cannot be read: {e}"));
    eprintln!("{}", deny_line(&req.method, &req.path, &peer));
    send_error(
        w,
        req,
        &ApiError {
            retry_after: false,
            code: 401,
            kind: "authentication_error",
            message: "Invalid API Key".to_owned(),
        },
    )?;
    Ok(true)
}

/// Dispatches one request; `Ok(true)` when the connection may carry another.
fn route(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    if key_denied(&state.api_keys, req, w)? {
        return Ok(true);
    }
    let r = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health" | "/v1/health") => return health(state, req, w),
        ("GET", "/v1/models" | "/models") => Ok(models(state)),
        ("GET", "/props") => Ok(props(state)),
        ("GET", "/slots") => Ok(slots(state)),
        ("GET", "/metrics") => return metrics(state, req, w),
        ("POST", p) if p.starts_with("/slots/") => {
            return slot_action(state, req, w, &p["/slots/".len()..]);
        }
        ("POST", "/residency/reset") => return residency_reset(state, req, w),
        ("POST", "/shutdown") => return post_shutdown(state, req, w),
        ("POST", "/completion" | "/completions") => return completion(state, req, w),
        ("POST", "/v1/completions") => return oaicompl::completions(state, req, w),
        ("POST", "/v1/chat/completions" | "/chat/completions") => return chat(state, req, w),
        ("POST", "/v1/chat/completions/input_tokens" | "/chat/completions/input_tokens") => {
            return oaicompl::count_tokens(state, req, w);
        }
        ("POST", "/v1/messages") => return anthropic::messages(state, req, w),
        ("POST", "/v1/messages/count_tokens") => return anthropic::count_tokens(state, req, w),
        ("POST", "/v1/responses" | "/responses") => return responses::create(state, req, w),
        ("POST", "/v1/responses/input_tokens" | "/responses/input_tokens") => {
            return responses::input_tokens(state, req, w);
        }
        ("POST", "/tokenize") => body(req).and_then(|b| tokenize(state, &b)),
        ("POST", "/detokenize") => body(req).and_then(|b| detokenize(state, &b)),
        ("POST", "/apply-template") => body(req).and_then(|b| apply_template(state, &b)),
        ("OPTIONS", _) => {
            http::respond(w, req, 204, "text/plain", &cors_preflight(), b"")?;
            return Ok(true);
        }
        _ => Err(ApiError {
            retry_after: false,
            code: 404,
            kind: "not_found_error",
            message: "File Not Found".to_owned(),
        }),
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

// ---------------------------------------------------------------- field helpers

fn get_f(o: &Map<String, Value>, k: &str) -> Option<f64> {
    o.get(k).and_then(Value::as_f64)
}

/// An integer field. Absent, `null` or not a number is `None`; a number that is
/// not an integer (`2.5`, or past `i64`) is a 400 naming the field, where
/// llama-server would truncate it. An integer-valued float (`2.0`) passes.
fn get_i(o: &Map<String, Value>, k: &str) -> Result<Option<i64>, ApiError> {
    let Some(v) = o.get(k).filter(|v| v.is_number()) else {
        return Ok(None);
    };
    if let Some(i) = v.as_i64() {
        return Ok(Some(i));
    }
    match v.as_f64() {
        Some(f) if f.fract() == 0.0 && f >= i64::MIN as f64 && f < i64::MAX as f64 => {
            Ok(Some(f as i64))
        }
        _ => Err(invalid(format!("{k} must be an integer, not {v}"))),
    }
}

fn get_b(o: &Map<String, Value>, k: &str) -> Option<bool> {
    o.get(k).and_then(Value::as_bool)
}

/// `stop` as a string or an array of strings; an array element that is not a
/// string is a 400 naming the field.
fn stop_list(v: Option<&Value>) -> Result<Vec<String>, ApiError> {
    match v {
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid(format!("stop must hold strings, not {x}")))
            })
            .collect(),
        _ => Ok(Vec::new()),
    }
}

/// Knobs shared by both generation endpoints. `n_predict` wins over the OpenAI names.
/// `cache_prompt` (default `true`, as llama-server) keeps the longest prefix the
/// engine's cache already holds; `false` resets it first.
///
/// A field this server cannot honor is a 400 naming it, never a 200 that ignores it:
/// `n_probs > 0`, `response_format` other than `{"type":"text"}`, `json_schema`, a
/// non-empty `grammar`, `logprobs` other than `false` (a bool, or OpenAI's text
/// completion count), `top_logprobs > 0`, `n > 1`, `tool_choice` other than
/// `"none"` or `"auto"` (nothing forces a call without a grammar), and a
/// non-empty `logit_bias`. `null` counts as absent. Other unknown fields are
/// ignored. The two routes whose answers render probabilities take their
/// fields out of the body before this runs ([`completion_plan_logprobs`],
/// [`chat_plan_logprobs`]); every other route keeps the refusal.
///
/// The penalties take llama-server's names and defaults (`repeat_penalty` 1,
/// `frequency_penalty` 0, `presence_penalty` 0, `repeat_last_n` 64) and
/// llama.cpp's checks (`common_sampler_init`): a non-finite penalty, or a
/// `repeat_penalty` that is not `> 0` with a finite reciprocal, is a 400
/// naming it. `repeat_last_n` outside llama-server's `0..=i32::MAX` is a 400. A
/// greedy request takes the engine's argmax, which no penalty reaches, so
/// penalties that change a logit with `temperature <= 0` are a 400.
fn gen_params(state: &State, o: &Map<String, Value>) -> Result<GenParams, ApiError> {
    let set = |k: &str| o.get(k).filter(|v| !v.is_null());
    let refused = [
        ("n_probs", get_i(o, "n_probs")?.is_some_and(|n| n > 0)),
        (
            "response_format",
            set("response_format")
                .is_some_and(|v| v.get("type").and_then(Value::as_str) != Some("text")),
        ),
        ("json_schema", set("json_schema").is_some()),
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
        (
            "tool_choice",
            set("tool_choice").is_some_and(|v| !matches!(v.as_str(), Some("none" | "auto"))),
        ),
        (
            "logit_bias",
            set("logit_bias").is_some_and(|v| match v {
                Value::Array(a) => !a.is_empty(),
                Value::Object(m) => !m.is_empty(),
                _ => true,
            }),
        ),
    ];
    if let Some((field, _)) = refused.iter().find(|(_, hit)| *hit) {
        return Err(invalid(format!("{field} is not supported by this server")));
    }
    let d = SamplingParams::default();
    let seed = match get_i(o, "seed")? {
        Some(s) if s >= 0 => s as u64,
        _ => mix(state.next_id() ^ unix_now().rotate_left(17)) & u64::from(u32::MAX),
    };
    let n_predict = get_i(o, "n_predict")?
        .or(get_i(o, "max_tokens")?)
        .or(get_i(o, "max_completion_tokens")?)
        .unwrap_or(-1);
    let top_k = get_i(o, "top_k")?;
    let penalty = |k: &str, d: f32, ok: fn(f32) -> bool, rule: &str| match set(k) {
        None => Ok(d),
        Some(v) => v
            .as_f64()
            .map(|x| x as f32)
            .filter(|&x| ok(x))
            .ok_or_else(|| invalid(format!("{k} must be {rule}, not {v}"))),
    };
    let finite = "a finite number";
    let repeat_last_n = match (set("repeat_last_n"), get_i(o, "repeat_last_n")?) {
        (None, _) => d.repeat_last_n,
        (Some(v), n) => n
            .filter(|&n| (0..=i64::from(i32::MAX)).contains(&n))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                invalid(format!(
                    "repeat_last_n must be an integer in 0..={}, as llama-server takes it, not {v}",
                    i32::MAX
                ))
            })?,
    };
    let sampling = SamplingParams {
        temperature: get_f(o, "temperature").map_or(d.temperature, |t| t as f32),
        top_k: top_k.map_or(d.top_k, |k| i32::try_from(k).unwrap_or(i32::MAX)),
        top_p: get_f(o, "top_p").map_or(d.top_p, |t| t as f32),
        min_p: get_f(o, "min_p").map_or(d.min_p, |t| t as f32),
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
    let p = GenParams {
        n_predict: if n_predict < 0 { -1 } else { n_predict },
        sampling,
        stop: stop_list(o.get("stop"))?,
        ignore_eos: get_b(o, "ignore_eos").unwrap_or(false),
        stream: get_b(o, "stream").unwrap_or(false),
        timings_per_token: get_b(o, "timings_per_token").unwrap_or(false),
        return_progress: get_b(o, "return_progress").unwrap_or(false),
        include_usage: o
            .get("stream_options")
            .and_then(|s| s.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        cache_prompt: get_b(o, "cache_prompt").unwrap_or(true),
        reasoning_budget: reasoning_budget(o)?,
        think_entry: ThinkEntry::Closed,
        logprobs: None,
    };
    Ok(p)
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
        ))),
        None => Err(invalid(format!(
            "reasoning_budget must be an integer, not {}",
            json_type(v)
        ))),
    }
}

/// A JSON value's type, for a field's refusal naming what it got.
fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// The think-span budget lives only where a span can open: a prompt that
/// ends with the start tag starts the model inside it, one that ends with
/// the closed span (the template with thinking off) has no reasoning to cap,
/// so the budget is silently ignored there — llama-server ignores
/// `--reasoning-budget` when thinking is already off by other means — and one
/// that ends with neither leaves the span to the model, where the budget
/// starts counting at the start tag, as llama-server's sampler does. The
/// entry lands in `p` for the generation's budget tracker whatever the
/// budget: the ids before a model-opened span spend nothing, and its close
/// ids are not forced until the span opens.
fn gate_reasoning_budget(p: &mut GenParams, prompt: &str) {
    p.think_entry = ThinkEntry::of_prompt(prompt);
    if p.reasoning_budget.is_some() && p.think_entry == ThinkEntry::Closed {
        p.reasoning_budget = None;
    }
}

/// llama-server's `generation_settings`, with the values this server applies.
/// Knobs it does not implement are reported at their neutral setting.
fn generation_settings(state: &State, p: &GenParams) -> Value {
    json!({
        "n_ctx": state.info.ctx_max,
        "n_predict": p.n_predict,
        "model": state.alias,
        "seed": p.sampling.seed,
        "temperature": p.sampling.temperature,
        "dynatemp_range": 0.0,
        "dynatemp_exponent": 1.0,
        "top_k": p.sampling.top_k,
        "top_p": p.sampling.top_p,
        "min_p": p.sampling.min_p,
        "tfs_z": 1.0,
        "typical_p": 1.0,
        "repeat_last_n": p.sampling.repeat_last_n,
        "repeat_penalty": p.sampling.repeat_penalty,
        "presence_penalty": p.sampling.presence_penalty,
        "frequency_penalty": p.sampling.frequency_penalty,
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
        "logit_bias": [],
        "n_probs": p.logprobs.as_ref().map_or(0, logprobs::Ask::asked),
        "min_keep": 0,
        "grammar": "",
        "samplers": ["penalties", "top_k", "top_p", "min_p", "temperature"],
    })
}

/// The settings a request with an empty body would get (seed shown as llama-server's default).
fn default_params() -> GenParams {
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
        think_entry: ThinkEntry::Closed,
        logprobs: None,
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

fn models(state: &State) -> Value {
    crate::models::listing(
        &state.alias,
        state.start_unix,
        json!({ "n_vocab": state.info.n_vocab, "n_ctx_train": state.info.ctx_max }),
        Some(state.info.ctx_max),
    )
}

fn props(state: &State) -> Value {
    json!({
        "system_prompt": "",
        "model_alias": state.alias,
        "model_path": state.model_path,
        "model_name": state.alias,
        "default_generation_settings": generation_settings(state, &default_params()),
        "total_slots": state.parallel,
        "chat_template": state.template.source(),
        "chat_template_caps": {},
        "bos_token": state.info.bos_text,
        "eos_token": state.info.eos_text,
        "modalities": { "vision": state.media.is_some(), "audio": false },
        "n_ctx": state.info.ctx_max,
        "build_info": format!("bloomery-serve {VERSION}"),
        "engine": state.engine_props,
    })
}

/// `engine.version`: the crate version and the commit the build script found
/// (`unknown` for a tree without git).
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
                _ => generation_settings(state, &default_params()),
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
        let n_past: usize = b.slots().iter().map(|s| s.view.n_past).sum();
        (state.parallel - b.free(), b.waiting(), n_past)
    };
    let room = state.info.ctx_max * state.parallel;
    let kv_ratio = if room > 0 {
        n_past as f64 / room as f64
    } else {
        0.0
    };
    let turns = relock(&state.shared.turns).is_some();
    let text = {
        let s = relock(&state.shared.stats);
        // Throughput since start: the ratio of the matching `*_total` counters, so a
        // scrape never changes it and any other window is `rate()` on those counters.
        let rate = |n: u64, ms: f64| {
            if n > 0 && ms > 0.0 {
                1e3 / ms * n as f64
            } else {
                0.0
            }
        };
        let per_decode = if s.n_decode_total > 0 {
            s.n_busy_slots_total as f64 / s.n_decode_total as f64
        } else {
            0.0
        };
        let rows: [(&str, &str, &str, String); 18] = [
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
                rate(s.n_prompt_total, s.t_prompt_ms_total).to_string(),
            ),
            (
                "gauge",
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s.",
                rate(s.n_predicted_total, s.t_predicted_ms_total).to_string(),
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
        let mut out = String::new();
        for (kind, name, help, value) in rows.into_iter().chain(swap.into_iter().filter(|_| turns))
        {
            out.push_str(&format!(
                "# HELP llamacpp:{name} {help}\n# TYPE llamacpp:{name} {kind}\nllamacpp:{name} {value}\n"
            ));
        }
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

// ---------------------------------------------------------------- tokenizer endpoints

fn tokenize(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    let content = b.get("content").and_then(Value::as_str).unwrap_or("");
    let add_special = get_b(b, "add_special").unwrap_or(false);
    let with_pieces = get_b(b, "with_pieces").unwrap_or(false);
    let t = &*state.tok;
    let mut ids = Vec::new();
    if add_special && t.add_bos() {
        ids.push(t.bos());
    }
    ids.extend(t.encode(content));
    let tokens: Vec<Value> = if with_pieces {
        ids.iter()
            .map(|&id| json!({ "id": id, "piece": t.decode(&[id]) }))
            .collect()
    } else {
        ids.iter().map(|&id| json!(id)).collect()
    };
    Ok(json!({ "tokens": tokens }))
}

fn detokenize(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    let ids = match b.get("tokens") {
        None => Vec::new(),
        Some(v) => token_ids(state, v)?,
    };
    Ok(json!({ "content": state.tok.decode(&ids) }))
}

fn token_ids(state: &State, v: &Value) -> Result<Vec<u32>, ApiError> {
    let Value::Array(a) = v else {
        return Err(invalid("tokens must be an array of token ids"));
    };
    a.iter()
        .map(|x| {
            x.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < state.info.n_vocab)
                .ok_or_else(|| invalid(format!("invalid token id {x}")))
        })
        .collect()
}

fn apply_template(state: &State, b: &Map<String, Value>) -> Result<Value, ApiError> {
    // The unexpanded placeholder: an image is one placeholder token in the
    // render, and only the request that runs it expands it to its span.
    Ok(json!({ "prompt": render_chat(state, b)?.text }))
}

/// A chat template's render: its text, each image as the model's
/// placeholder, and the images' URLs in the order of their placeholders.
struct Rendered {
    text: String,
    images: Vec<String>,
}

/// Renders the chat template for an OpenAI `messages` body. An array content
/// becomes one text: on an engine that takes images, its parts flattened by
/// the model's rule ([`media::flatten`]), each image the model's placeholder;
/// on one that takes none, its text parts joined, any other part refused by
/// name ([`content_text`]). Nothing is dropped. A `developer` message reaches
/// the template as `system`, as llama-server maps it, and the thinking vars
/// ([`thinking_vars`]) reach it in llama-server's order.
fn render_chat(state: &State, b: &Map<String, Value>) -> Result<Rendered, ApiError> {
    let Some(Value::Array(msgs)) = b.get("messages") else {
        return Err(invalid("'messages' is required and must be an array"));
    };
    // GPT-OSS's template is the one that renders the `developer` role itself,
    // named by the `<|channel|>` marker in its source.
    let gpt_oss = state.template.source().contains("<|channel|>");
    let mut messages = Vec::with_capacity(msgs.len());
    let mut images = Vec::new();
    for m in msgs {
        let Value::Object(m) = m else {
            return Err(invalid("each message must be an object"));
        };
        let Some(role) = m.get("role").and_then(Value::as_str) else {
            return Err(invalid("each message needs a string 'role'"));
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
        if let Some(Value::Array(parts)) = m.get("content") {
            let text = match &state.media {
                None => content_text(parts)?,
                Some(model) => {
                    let f = media::flatten(role, parts, &**model).map_err(|e| media_error(&e))?;
                    images.extend(f.images.into_iter().map(str::to_owned));
                    f.text
                }
            };
            flat.insert("content".into(), Value::String(text));
        }
        messages.push(Value::Object(flat));
    }
    let mut vars = Map::new();
    vars.insert("messages".into(), Value::Array(messages));
    vars.insert(
        "add_generation_prompt".into(),
        Value::Bool(get_b(b, "add_generation_prompt").unwrap_or(true)),
    );
    vars.insert(
        "bos_token".into(),
        Value::String(state.info.bos_text.clone()),
    );
    vars.insert(
        "eos_token".into(),
        Value::String(state.info.eos_text.clone()),
    );
    if let Some(v) = b.get("tools").filter(|v| !v.is_null()) {
        vars.insert("tools".into(), v.clone());
    }
    thinking_vars(b, &mut vars)?;
    let text = state.template.render(&vars).map_err(|e| ApiError {
        retry_after: false,
        code: 500,
        kind: "server_error",
        message: e.to_string(),
    })?;
    Ok(Rendered { text, images })
}

/// The thinking vars a chat body carries, in llama-server's order: the kwargs
/// `enable_thinking` first — a bool sets it, a string is a 400 naming it —
/// then the OpenAI `reasoning_effort`, whose `"none"` turns thinking off and
/// does not reach the template as itself (a kwargs `reasoning_effort` is
/// erased with it). The resolved `enable_thinking` replaces the kwargs entry,
/// as llama-server's template application writes the resolved value over it,
/// so `"none"` wins over a kwargs `true`. It lands in `vars` only when
/// something asked: a template that tells an undefined `enable_thinking` from
/// `true` (V4.1's takes its `thinking` from it) keeps its own default when
/// nothing did.
fn thinking_vars(b: &Map<String, Value>, vars: &mut Map<String, Value>) -> Result<(), ApiError> {
    let kwargs = b.get("chat_template_kwargs").and_then(Value::as_object);
    let mut thinking: Option<bool> = None;
    if let Some(kw) = kwargs {
        match kw.get("enable_thinking") {
            Some(Value::Bool(on)) => thinking = Some(*on),
            Some(Value::String(_)) => {
                return Err(invalid(
                    "invalid type for \"enable_thinking\" (expected boolean, got string)",
                ));
            }
            _ => {}
        }
    }
    let effort = b.get("reasoning_effort").filter(|v| !v.is_null());
    let off = effort.and_then(Value::as_str) == Some("none");
    if off {
        thinking = Some(false);
    } else if let Some(v) = effort {
        vars.insert("reasoning_effort".into(), v.clone());
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
                        && !(off && k.as_str() == "reasoning_effort")
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
            code: 503,
            kind: "unavailable_error",
            message: format!(
                "every one of the {} slots is busy and {depth} requests wait, the queue's depth",
                state.parallel
            ),
        },
        Refusal::Busy => ApiError {
            retry_after: false,
            code: 503,
            kind: "unavailable_error",
            message: format!("{what} is processing a request"),
        },
        Refusal::Dead(reason) => dead_engine(&reason),
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

/// A refusal of the media part: the request's 400, but for a prepare whose
/// span is empty, the model's 500.
fn media_error(e: &MediaError) -> ApiError {
    match e {
        MediaError::EmptySpan(_) => engine_error(e),
        _ => invalid(e.to_string()),
    }
}

/// The prompt of a rendered chat: its ids, and on an engine that takes images
/// each image expanded to its span with its feed ([`media::expand_prompt`]),
/// which also refuses a placeholder no image part made.
fn chat_prompt(state: &State, rendered: &Rendered) -> Result<Prompt, ApiError> {
    let ids = state.tok.encode(&rendered.text);
    let Some(model) = &state.media else {
        return Ok(Prompt::from(ids));
    };
    let urls: Vec<&str> = rendered.images.iter().map(String::as_str).collect();
    media::expand_prompt(&ids, &urls, &**model).map_err(|e| media_error(&e))
}

/// Validates the prompt, hands the request to the engine thread, and passes
/// its events to `sink` with the slot it took. `input` is what the engine is
/// fed: the ids, and the images' spans and feeds of a chat that carries any.
/// Returns the outcome and the slot; an error before any event is the
/// request's answer. A sink that fails ends the request: the engine thread
/// sees its channel closed.
fn run_gen(
    state: &State,
    input: &Prompt,
    prompt: Value,
    p: &GenParams,
    sink: &mut dyn FnMut(Event<'_>, usize) -> io::Result<()>,
) -> Result<(Result<Outcome, GenError>, usize), ApiError> {
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
    if ids.len() >= state.info.ctx_max {
        return Err(ApiError {
            retry_after: false,
            code: 400,
            kind: "exceed_context_size_error",
            message: format!(
                "the prompt has {} tokens and the context holds {}",
                ids.len(),
                state.info.ctx_max
            ),
        });
    }
    let (events, rx) = mpsc::channel();
    let submit = Submit {
        p: p.clone(),
        prompt,
        settings: generation_settings(state, p),
        media: input.feeds.clone(),
        events,
    };
    relock(&state.shared.board)
        .enqueue(input.held.clone(), submit)
        .map_err(|r| refused(state, r, "the slot"))?;
    state.shared.work.notify_one();
    let mut slot = None;
    loop {
        let Ok(m) = rx.recv() else {
            return Err(engine_gone(state, slot.is_some()));
        };
        let at = slot.unwrap_or(0);
        let sent = match m {
            Msg::Started(s) => {
                slot = Some(s);
                Ok(())
            }
            Msg::Prompt(t) => sink(Event::Prompt(&t), at),
            Msg::Text(text, t) => sink(Event::Text(&text, &t), at),
            Msg::Done(r) => return Ok((r, at)),
            Msg::Refused(why) => {
                return Err(ApiError {
                    retry_after: true,
                    code: 503,
                    kind: "unavailable_error",
                    message: why,
                });
            }
        };
        if let Err(e) = sent {
            return Ok((Err(GenError::Client(e)), at));
        }
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
            .ok_or_else(|| invalid("'filename' is required"))?
            .as_str()
            .ok_or_else(|| invalid("'filename' must be a string"))?;
        if !slotfile::valid_filename(f) {
            return Err(invalid("Invalid filename"));
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
                code: 501,
                kind: "not_supported_error",
                message: e.to_string(),
            }),
            None,
        ),
        StateError::Engine(e) => (Err(engine_error(&e)), Some(e)),
        StateError::Format(_) | StateError::Io(_) => (
            Err(ApiError {
                retry_after: false,
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
    if let Some(model) = &state.media {
        media::expand_prompt(&ids, &[], &**model).map_err(|e| media_error(&e))?;
    }
    Ok(ids)
}

fn completion_ids(state: &State, v: Option<&Value>) -> Result<Vec<u32>, ApiError> {
    let e = &*state.tok;
    let no_media =
        || invalid("Multimodal data provided, but model does not support multimodal requests.");
    match v {
        Some(o) if carries_media(o) => Err(no_media()),
        Some(Value::String(s)) => {
            let mut ids = Vec::new();
            if e.add_bos() {
                ids.push(e.bos());
            }
            ids.extend(e.encode(s));
            Ok(ids)
        }
        Some(Value::Array(a)) => {
            let mut ids = Vec::new();
            if a.first().is_some_and(Value::is_string) && e.add_bos() {
                ids.push(e.bos());
            }
            for x in a {
                match x {
                    Value::String(s) => ids.extend(e.encode(s)),
                    o if carries_media(o) => return Err(no_media()),
                    other => ids.extend(token_ids(state, &Value::Array(vec![other.clone()]))?),
                }
            }
            Ok(ids)
        }
        _ => Err(invalid("'prompt' must be a string or an array of tokens")),
    }
}

/// Whether a prompt object carries media: a `multimodal_data` that holds
/// anything (`null` or an empty array, string or object holds nothing).
fn carries_media(v: &Value) -> bool {
    match v.get("multimodal_data") {
        None | Some(Value::Null) => false,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(_) => true,
    }
}

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
/// the prompt's ids ([`completion_prompt`]) and the think-span budget, whose
/// span check reads the prompt's text (a string prompt is its own, an id
/// array's its decode). Every route that runs a completion builds its
/// generation here.
fn completion_plan(state: &State, b: &Map<String, Value>) -> Result<CompletionPlan, ApiError> {
    let mut p = gen_params(state, b)?;
    let ids = completion_prompt(state, b.get("prompt"))?;
    if p.reasoning_budget.is_some() {
        let text = match b.get("prompt") {
            Some(Value::String(s)) => s.clone(),
            _ => state.tok.decode(&ids),
        };
        gate_reasoning_budget(&mut p, &text);
    }
    Ok(CompletionPlan {
        p,
        input: Prompt::from(ids),
        prompt: b.get("prompt").cloned().unwrap_or(Value::Null),
        return_tokens: get_b(b, "return_tokens").unwrap_or(false),
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
    let plan = match body(req).and_then(|mut b| completion_plan_logprobs(state, &mut b)) {
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
        return match run_gen(state, &input, prompt.clone(), &p, &mut |_, _| Ok(())) {
            Err(e) => send_error(w, req, &e),
            Ok((Err(e), _)) => send_error(w, req, &engine_error(&e)),
            Ok((Ok(o), slot)) => send_json(
                w,
                req,
                200,
                &completion_final(state, &o, &p, &prompt, return_tokens, slot),
            ),
        };
    }
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
                Event::Text(text, t) => {
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
        run_gen(state, &input, prompt.clone(), &p, &mut sink)
    };
    let slot = r.as_ref().map_or(0, |(_, slot)| *slot);
    finish_stream(req, stream, w_opt, r.map(|(o, _)| o), |s, o| {
        sse(
            s,
            &completion_final(state, o, &p, &prompt, return_tokens, slot),
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
fn finish_stream(
    req: &Request,
    stream: Option<EventStream<'_>>,
    w_opt: Option<&mut TcpStream>,
    r: Result<Result<Outcome, GenError>, ApiError>,
    last: impl FnOnce(&mut EventStream<'_>, &Outcome) -> io::Result<()>,
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
            sse(
                &mut stream,
                &json!({ "error": error_body(err.code, err.kind, &err.message)["error"] }),
            )?;
        }
        Err(e) => sse(
            &mut stream,
            &json!({ "error": error_body(e.code, e.kind, &e.message)["error"] }),
        )?,
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
    /// content, then one delta per tool call.
    fn deltas(&self, d: &Message) -> Vec<Value> {
        let mut out = Vec::new();
        if !d.reasoning.is_empty() {
            out.push(json!({ "reasoning_content": d.reasoning }));
        }
        if !d.content.is_empty() {
            out.push(json!({ "content": d.content }));
        }
        for c in &d.calls {
            let mut call = self.tool_call(c);
            call["index"] = json!(c.index);
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

/// Whether the output is scanned for tool calls: non-empty `tools` and a
/// `tool_choice` other than `"none"` (the template still sees the tools with
/// `"none"`).
fn parses_tools(b: &Map<String, Value>) -> bool {
    b.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
        && b.get("tool_choice").and_then(Value::as_str) != Some("none")
}

/// The scan a chat request's output gets: none, or the markup of the
/// server's template; a request that asks for tool calls under a template
/// whose markup no parser reads is refused, never answered with the calls
/// left in `content`.
fn tool_scan(state: &State, b: &Map<String, Value>) -> Result<Option<Tools>, ApiError> {
    if !parses_tools(b) {
        return Ok(None);
    }
    Ok(Some(match state.tool_format {
        ToolFormat::Dsml => Tools::Dsml,
        ToolFormat::GlmXml => Tools::GlmXml(ArgTypes::of_tools(b.get("tools"))),
        ToolFormat::Hermes => Tools::Hermes,
        ToolFormat::QwenXml => Tools::QwenXml(ParamKinds::of_tools(b.get("tools"))),
        ToolFormat::Unparsed => {
            return Err(ApiError {
                retry_after: false,
                code: 501,
                kind: "not_supported_error",
                message: "the chat template's tool-call markup has no parser in this server \
                          (DSML, GLM's, Hermes' and Qwen's XML are parsed): send the request \
                          without tools or with tool_choice \"none\""
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
        code: 500,
        kind: "server_error",
        message: format!("tool-call markup: {e}"),
    }
}

fn usage(o: &Outcome) -> Value {
    json!({
        "completion_tokens": o.timings.predicted_n,
        "prompt_tokens": o.timings.n_prompt,
        "total_tokens": o.timings.predicted_n + o.timings.n_prompt,
        "prompt_tokens_details": { "cached_tokens": o.timings.cache_n },
    })
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
/// tool-call scan and the think-span budget. The chat path and every API
/// converted onto it build their generation here.
fn chat_plan(
    state: &State,
    b: &Map<String, Value>,
    format: Option<&Value>,
) -> Result<ChatPlan, ApiError> {
    let mut p = gen_params(state, b)?;
    let format = ReasoningFormat::from_request(format).map_err(invalid)?;
    let (text, input) = chat_input(state, b)?;
    let tools = tool_scan(state, b)?;
    gate_reasoning_budget(&mut p, &text);
    let parser = ChatParser::with_tools(&text, format, tools);
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
    let parsed = body(req).and_then(|mut b| {
        let plan = chat_plan_logprobs(state, &mut b)?;
        Ok((b, plan))
    });
    let (b, plan) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let ChatPlan {
        p,
        input,
        prompt,
        mut parser,
    } = plan;
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
        return match run_gen(state, &input, prompt, &p, &mut |_, _| Ok(())) {
            Err(e) => send_error(w, req, &e),
            Ok((Err(e), _)) => send_error(w, req, &engine_error(&e)),
            Ok((Ok(o), _)) => match parser
                .try_push(&o.content)
                .and_then(|_| parser.try_finish())
            {
                Ok(_) => {
                    let lp = p
                        .logprobs
                        .as_ref()
                        .map(|a| logprobs::chat_object(a.whole(vocab)));
                    send_json(
                        w,
                        req,
                        200,
                        &chat_final(&ids_meta, &o, parser.message(), lp),
                    )
                }
                Err(e) => send_error(w, req, &tool_markup_error(&e)),
            },
        };
    }
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    // Markup that does not parse stops the generation (the sink fails) and
    // ends the stream with an error event instead of a dropped connection.
    let mut markup: Option<MarkupError> = None;
    // The tokens whose entries went out: a chunk's last delta carries the
    // tokens its event counts past them; an event that makes no delta leaves
    // its tokens to the next chunk that has one, the last chunk what is left.
    let mut sent = 0;
    let r = {
        let meta = &ids_meta;
        let mut sink = |ev: Event<'_>, _slot: usize| -> io::Result<()> {
            if stream.is_none() {
                let w = w_opt
                    .take()
                    .ok_or_else(|| io::Error::other("stream writer taken"))?;
                let mut s = EventStream::start(w, req, 200, "text/event-stream")?;
                // OpenAI opens every stream with the role.
                sse(
                    &mut s,
                    &meta.chunk(json!([{
                        "finish_reason": null, "index": 0,
                        "delta": { "role": "assistant", "content": null },
                    }])),
                )?;
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
                Event::Text(text, t) => {
                    let d = match parser.try_push(text) {
                        Ok(d) => d,
                        Err(e) => {
                            markup = Some(e);
                            return Err(io::Error::other("tool-call markup does not parse"));
                        }
                    };
                    let deltas = meta.deltas(&d);
                    let n = deltas.len();
                    for (i, delta) in deltas.into_iter().enumerate() {
                        let mut v = meta.chunk(json!([{
                            "finish_reason": null, "index": 0, "delta": delta,
                        }]));
                        if i + 1 == n
                            && let Some(ask) = &p.logprobs
                        {
                            let entries = ask.entries(sent, t.predicted_n, vocab);
                            sent = t.predicted_n;
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
        run_gen(state, &input, prompt, &p, &mut sink).map(|(o, _)| o)
    };
    let r = match markup {
        Some(e) => Err(tool_markup_error(&e)),
        None => r,
    };
    let include_usage = p.include_usage;
    let opened = stream.is_some();
    finish_stream(req, stream, w_opt, r, |s, o| {
        if !opened {
            sse(
                s,
                &ids_meta.chunk(json!([{
                    "finish_reason": null, "index": 0,
                    "delta": { "role": "assistant", "content": null },
                }])),
            )?;
        }
        let d = match parser.try_finish() {
            Ok(d) => d,
            Err(e) => {
                let e = tool_markup_error(&e);
                return sse(
                    s,
                    &json!({ "error": error_body(e.code, e.kind, &e.message)["error"] }),
                );
            }
        };
        // The tokens no chunk carried yet go with the last delta, or with the
        // finishing chunk when the parser releases none.
        let mut left = p
            .logprobs
            .as_ref()
            .map(|a| a.entries(sent, o.timings.predicted_n, vocab))
            .filter(|e| !e.is_empty());
        let deltas = ids_meta.deltas(&d);
        let n = deltas.len();
        for (i, delta) in deltas.into_iter().enumerate() {
            let mut v =
                ids_meta.chunk(json!([{ "finish_reason": null, "index": 0, "delta": delta }]));
            if i + 1 == n
                && let Some(e) = left.take()
            {
                v["choices"][0]["logprobs"] = logprobs::chat_object(e);
            }
            sse(s, &v)?;
        }
        let mut last = ids_meta.chunk(json!([{
            "finish_reason": chat_finish_reason(o, parser.message()), "index": 0, "delta": {},
        }]));
        if let Some(e) = left.take() {
            last["choices"][0]["logprobs"] = logprobs::chat_object(e);
        }
        if include_usage {
            sse(s, &last)?;
            last = ids_meta.chunk(json!([]));
            last["usage"] = usage(o);
        }
        last["timings"] = o.timings.to_json();
        sse(s, &last)?;
        s.send(b"data: [DONE]\n\n")
    })
}

/// The non-stream response. `reasoning_content` and `tool_calls` appear only when
/// non-empty, as llama-server writes them; `logprobs`, the chat path's
/// probabilities object, only when the request asked for it.
fn chat_final(meta: &ChatIds, o: &Outcome, m: &Message, probs: Option<Value>) -> Value {
    let mut message = json!({ "role": "assistant", "content": m.content });
    if !m.reasoning.is_empty() {
        message["reasoning_content"] = json!(m.reasoning);
    }
    if !m.calls.is_empty() {
        message["tool_calls"] = m.calls.iter().map(|c| meta.tool_call(c)).collect();
    }
    let mut choice = json!({
        "finish_reason": chat_finish_reason(o, m),
        "index": 0,
        "message": message,
    });
    if let Some(lp) = probs {
        choice["logprobs"] = lp;
    }
    json!({
        "choices": [choice],
        "created": meta.created,
        "model": meta.model,
        "object": "chat.completion",
        "usage": usage(o),
        "id": meta.id,
        "timings": o.timings.to_json(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        ACCEPT_BACKOFF, EBADF, EFAULT, EINVAL, EMFILE, ENFILE, End, Engine, EngineProps, Park,
        ServeError, SlotConfig, SlotQueue, Stop, after_accept_error, argv_redacted, carries_media,
        check_slots, content_text, deny_line, engine_object, id_half, shutdown_allowed,
    };
    use crate::engine::{DeviceProps, PlacementProps};
    use crate::flag::ApiKeys;
    use serde_json::{Value, json};
    use std::io;
    use std::net::SocketAddr;
    use std::time::Duration;

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

    /// `POST /shutdown` over a real socket: the 200 with its body reaches the
    /// client first, the end channel carries the shutdown naming a loopback
    /// peer, a second post answers 200 and changes nothing, and a generation
    /// request after it is a 503 naming the shutdown (pinned: the listener
    /// stands while the stop runs — the exit belongs to `run`, which no test
    /// enters — so the refusal the client reads is the gate's 503, not a
    /// refused connection). The engine thread leaves its loop on its own; its
    /// end, the engine's Drop, is the process exit's.
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
    /// as llama-server's pre-routing handler answers it before the check.
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

        let line = deny_line("POST", "/completion", "127.0.0.1:54094");
        assert!(
            line.contains("POST /completion") && line.contains("127.0.0.1:54094"),
            "{line}"
        );
        assert!(
            !line.contains(KEY),
            "the refusal's line carries no key: {line}"
        );
    }

    /// A key never reaches what the process itself prints or serves: the
    /// real binary started with `--api-key` — its startup line, the 401's
    /// stderr line, every other line it has printed, `/props` (whose
    /// `engine.args` is this process's own argv, the key in it, held back by
    /// [`argv_redacted`]) and a 400 error body carry none of the key's
    /// bytes. The real binary because the in-crate server prints no startup
    /// line and its argv is the test runner's, not a seat's.
    #[test]
    fn a_key_never_reaches_what_the_process_prints_or_serves() {
        let served = super::testserve::spawn_bin(&["--port", "0", "--api-key", KEY]);
        let ask = r#"{"prompt":"I believe the meaning of life is","max_tokens":4}"#;

        let (status, head, body) =
            super::testserve::roundtrip_head(served.addr, "POST", "/completion", &[], ask);
        assert_eq!(status, 401, "{body}");
        // A wrong key, whose bytes are the client's secret: they may reach
        // neither the answer nor a log line, exactly as the server's own key.
        const SENT: &str = "sk-a-wrong-key-the-client-sent";
        let (status, _head, body) = super::testserve::roundtrip_head(
            served.addr,
            "POST",
            "/completion",
            &[("Authorization", SENT)],
            ask,
        );
        assert_eq!(status, 401, "the wrong key: {body}");
        for text in [&head, &body] {
            assert!(!text.contains(KEY), "the 401 carries no key: {text}");
            assert!(!text.contains(SENT), "the 401 carries no sent key: {text}");
        }

        // The startup line, then the refusals' lines once they land.
        assert!(
            served.startup.contains("listening on"),
            "the startup line: {}",
            served.startup
        );
        assert!(
            !served.startup.contains(KEY),
            "the startup line carries no key"
        );
        let mut refusals = 0;
        while refusals < 2 {
            let line = served
                .lines
                .recv_timeout(Duration::from_secs(5))
                .expect("the refusals' stderr lines");
            assert!(!line.contains(KEY), "a printed line carries no key: {line}");
            assert!(
                !line.contains(SENT),
                "a printed line carries no sent key: {line}"
            );
            if line.contains("refused") {
                refusals += 1;
            }
        }

        for (method, path, headers, body_of, wants) in [
            ("GET", "/props", auth(BEARER), "", 200),
            ("POST", "/tokenize", auth(BEARER), "{ not json", 400),
        ] {
            let (status, _head, body) =
                super::testserve::roundtrip_head(served.addr, method, path, &headers, body_of);
            assert_eq!(status, wants, "{method} {path}: {body}");
            assert!(
                !body.contains(KEY),
                "{method} {path} carries no key: {body}"
            );
        }
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

        /// One body as the comparison takes it: JSON with `created` and
        /// `id` (the bind time and the wall-clock-mixed request id) pinned
        /// and `timings` dropped; a stream the same, chunk by chunk;
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
                let Value::Object(o) = v else { return };
                o.remove("timings");
                for leaf in ["created", "id"] {
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

    /// The orderly stop's wait on the engine thread — what it waits for (the
    /// loop's end, between engine calls) and what it does not (the engine's
    /// Drop, which can outlast it) — pinned at a small scale:
    /// [`ENGINE_STOP`]'s policy without spending its seconds. The servers
    /// start the way `run` starts them ([`Server::start`]); the engines are
    /// mocks owned by these gates alone.
    mod stop_wait {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        use crate::Engine;
        use crate::MockTokenizer;
        use crate::Tokenizer;
        use crate::api::{FATAL_LINGER, Server, ServerConfig, Shared, Stop, relock};
        use crate::genloop::Slot;
        use crate::sched::Reserve;
        use crate::worker::{Acted, Action};

        /// The waits' bound in these gates: past it, a call the stop arrives
        /// in is reported; the engine's Drop is made to outlast it.
        const BOUND: Duration = Duration::from_millis(400);

        /// An engine whose Drop sleeps `ms`, as a placed engine's does (a
        /// host tier's pinned pages freed, its threads joined), then notes
        /// that it finished: work that outlives the stop's bound with no call
        /// running.
        struct SlowDrop {
            ms: u64,
            dropped: Arc<AtomicBool>,
        }

        impl Engine for SlowDrop {
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
                "the slow-drop mock".to_owned()
            }
        }

        impl Drop for SlowDrop {
            fn drop(&mut self) {
                std::thread::sleep(Duration::from_millis(self.ms));
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

        /// An idle engine thread leaves its loop at once when the stop
        /// arrives, and the engine's Drop it leaves behind — however long —
        /// is not waited out: the wait ends under its bound while the Drop
        /// still runs.
        #[test]
        fn the_stop_waits_out_no_engine_drop() {
            let dropped = Arc::new(AtomicBool::new(false));
            let sh = started(Box::new(SlowDrop {
                ms: 900,
                dropped: Arc::clone(&dropped),
            }));
            sh.begin_stop(&Stop::Sigint);
            // A true under the bound is the loop's own leaving: the wait
            // wakes on its signal, not on the bound running out.
            assert!(
                sh.wait_loop_end(BOUND),
                "the loop left; only the engine's Drop was left to run"
            );
            assert!(
                !dropped.load(Ordering::SeqCst),
                "the engine's Drop was still running when the wait ended"
            );
            let until = Instant::now() + Duration::from_secs(5);
            while !dropped.load(Ordering::SeqCst) && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(
                dropped.load(Ordering::SeqCst),
                "the engine thread finished its engine's Drop"
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
            assert!(
                !sh.wait_loop_end(BOUND),
                "the loop was inside the call the stop arrived in"
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
    /// role set to `system`, on every fixture template: llama-server's
    /// mapping, which every template but GPT-OSS's gets. Each template
    /// renders the system message — none drops or refuses it — so the mapped
    /// role lands in the prompt's text.
    #[test]
    fn developer_messages_render_as_system_on_every_fixture_template() {
        let fixtures = [
            (
                "qwen38",
                include_str!("../tests/fixtures/qwen38-chat-template.jinja"),
            ),
            (
                "v41",
                include_str!("../tests/fixtures/v41-chat-template.jinja"),
            ),
            (
                "glm5",
                include_str!("../tests/fixtures/glm5-chat-template.jinja"),
            ),
            (
                "qwen3",
                include_str!("../tests/fixtures/qwen3-chat-template.jinja"),
            ),
            (
                "qwen36",
                include_str!("../tests/fixtures/qwen36-chat-template.jinja"),
            ),
        ];
        for (name, template) in fixtures {
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

    /// `reasoning_effort: "none"` on GLM-5's template renders as
    /// `chat_template_kwargs: {"enable_thinking": false}` does: no
    /// `reasoning_effort` reaches the template (its effort header keeps its
    /// default; it reads `reasoning_effort`, not `enable_thinking`), and
    /// another effort still reaches it.
    #[test]
    fn reasoning_effort_none_turns_thinking_off_on_glm5() {
        let (addr, _state, _ended) = super::testserve::spawn(
            Box::new(crate::MockEngine::new(64)),
            templated(include_str!("../tests/fixtures/glm5-chat-template.jinja")),
        );
        let messages = || json!([{ "role": "user", "content": "Hi" }]);
        let (status, off) = applied(
            addr,
            &json!({ "messages": messages(), "chat_template_kwargs": { "enable_thinking": false } }),
        );
        assert_eq!(status, 200, "{off}");
        let (status, none) = applied(
            addr,
            &json!({ "messages": messages(), "reasoning_effort": "none" }),
        );
        assert_eq!(status, 200, "{none}");
        assert_eq!(none, off, "none turns thinking off");
        let (status, high) = applied(
            addr,
            &json!({ "messages": messages(), "reasoning_effort": "high" }),
        );
        assert_eq!(status, 200, "{high}");
        assert_eq!(
            applied_prompt(&high),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>"
        );
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
    /// `reasoning_content`.
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
        assert!(
            off["content"].as_str().is_some_and(|s| !s.is_empty()),
            "thinking off: the whole output is content: {off}"
        );
    }
}
