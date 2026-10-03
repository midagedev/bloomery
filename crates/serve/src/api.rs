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

use std::fmt;
use std::io::{self, BufRead, BufReader, Read};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::dsml::{ChatParser, Message, ToolCall, ToolFormat, Tools};
use crate::engine::{
    DraftProps, Engine, EngineError, EngineProps, ModelProps, PlacementProps, SamplerFactory,
    SamplingParams, StateError, Tokenizer,
};
use crate::genloop::{self, Event, GenError, GenParams, Outcome, Slot, Timings, ms_since};
use crate::glmxml::{ArgTypes, GlmXmlError};
use crate::http::{self, EventStream, Request};
use crate::reasoning::ReasoningFormat;
use crate::sampling;
use crate::sched::{Board, Refusal, Reserve, SlotConfig, Use, default_depth};
use crate::slotfile;
use crate::template::{ChatTemplate, TemplateError};
use crate::worker::{self, Acted, Action, Msg, Shared, Submit};

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
    let rows = engine.advance_rows();
    if n > 1 && rows > 1 {
        return refuse(format!(
            "--parallel {n}: this engine drafts ({rows} rows a pass), and a step of several \
             slots does not draft"
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
    /// ([`Engine::slots`]), several slots on an engine that drafts, and a
    /// depth no queue can reach are refused by name ([`ServeError::Slots`]).
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
            tool_format: ToolFormat::of_template(template.source()),
            template,
            alias: config.model_alias,
            model_path: config.model_path,
            engine_props,
            slot_save_path: config.slot_save_path,
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
    /// listener fails; returns the server's state and what ends it.
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
        let accept_state = Arc::clone(&state);
        let port = listener.local_addr().map_or(0, |a| a.port());
        let live = Arc::new(AtomicUsize::new(0));
        thread::spawn(move || {
            let mut repeats = 0;
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => {
                        repeats = 0;
                        admit(&accept_state, &live, port, stream);
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
                            let _ = accept_state.shared.end.send(End::Io(gone));
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
        Ok((state, ended))
    }

    /// Serves until the listener fails or the engine does, and returns why.
    pub fn run(self) -> ServeError {
        let (state, ended) = match self.start() {
            Ok(s) => s,
            Err(e) => return ServeError::Io(e),
        };
        match ended.recv() {
            Ok(End::Io(e)) => ServeError::Io(e),
            Ok(End::Engine(f)) => {
                thread::sleep(state.fatal_linger);
                ServeError::Engine(f)
            }
            // `state` holds a sender, so the channel cannot close while we wait.
            Err(mpsc::RecvError) => ServeError::Io(io::Error::other("accept loop vanished")),
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
    /// Where slot files live; `None` refuses slot actions.
    slot_save_path: Option<PathBuf>,
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
        format!(
            "{:016x}{:016x}",
            mix(t ^ n),
            mix(n.wrapping_add(0x5bd1_e995))
        )
    }

    fn fatal(&self) -> Option<String> {
        relock(&self.shared.fatal).clone()
    }
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
pub(crate) struct Permit(Arc<AtomicUsize>);

impl Permit {
    /// A place, or `None` when [`MAX_CONNECTIONS`] are live.
    pub(crate) fn take(live: &Arc<AtomicUsize>) -> Option<Permit> {
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

/// Serves `stream` on a thread of its own, named `serve:<port>`, or refuses it
/// with a 503 when [`MAX_CONNECTIONS`] are live or no thread can be started.
fn admit(state: &Arc<State>, live: &Arc<AtomicUsize>, port: u16, stream: TcpStream) {
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
    let state = Arc::clone(state);
    let spawned = thread::Builder::new()
        .name(format!("serve:{port}"))
        .spawn(move || {
            let _permit = permit;
            serve_conn(&state, stream);
        });
    if let Err(e) = spawned {
        refuse(answer, &format!("cannot start a connection thread: {e}"));
    }
}

/// Answers a connection the server does not serve with a 503 carrying
/// `Retry-After`, then closes it after reading what the client sent, for at most
/// about [`REFUSE_DRAIN`] twice.
pub(crate) fn refuse(mut stream: TcpStream, message: &str) {
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
pub(crate) fn next_request_arrives(r: &mut BufReader<TcpStream>) -> bool {
    if !r.buffer().is_empty() {
        return true;
    }
    if r.get_ref().set_read_timeout(Some(KEEP_ALIVE_IDLE)).is_err() {
        return false;
    }
    let arrived = r.fill_buf().is_ok_and(|b| !b.is_empty());
    arrived && r.get_ref().set_read_timeout(Some(REQUEST_READ)).is_ok()
}

fn serve_conn(state: &State, stream: TcpStream) {
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
        match route(state, &req, &mut w) {
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

/// Dispatches one request; `Ok(true)` when the connection may carry another.
fn route(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
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
        ("POST", "/completion" | "/completions") => return completion(state, req, w),
        ("POST", "/v1/chat/completions" | "/chat/completions") => return chat(state, req, w),
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
/// non-empty `grammar`, `logprobs: true`, `top_logprobs > 0`, `n > 1`, and
/// `tool_choice` other than `"none"` or `"auto"` (nothing forces a call without a
/// grammar). `null` counts as absent. Other unknown fields
/// are ignored.
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
        ("logprobs", get_b(o, "logprobs") == Some(true)),
        (
            "top_logprobs",
            get_i(o, "top_logprobs")?.is_some_and(|n| n > 0),
        ),
        ("n", get_i(o, "n")?.is_some_and(|n| n > 1)),
        (
            "tool_choice",
            set("tool_choice").is_some_and(|v| !matches!(v.as_str(), Some("none" | "auto"))),
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
    let p = GenParams {
        n_predict: if n_predict < 0 { -1 } else { n_predict },
        sampling: SamplingParams {
            temperature: get_f(o, "temperature").map_or(d.temperature, |t| t as f32),
            top_k: top_k.map_or(d.top_k, |k| i32::try_from(k).unwrap_or(i32::MAX)),
            top_p: get_f(o, "top_p").map_or(d.top_p, |t| t as f32),
            min_p: get_f(o, "min_p").map_or(d.min_p, |t| t as f32),
            seed,
        },
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
    };
    Ok(p)
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
        "repeat_last_n": 0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
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
        "n_probs": 0,
        "min_keep": 0,
        "grammar": "",
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
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
    }
}

// ---------------------------------------------------------------- read-only endpoints

fn health(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    if let Some(reason) = state.fatal() {
        let mut v = error_body(503, "unavailable_error", &reason);
        v["status"] = json!("error");
        v["reason"] = json!(reason);
        return send_json(w, req, 503, &v);
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
        "modalities": { "vision": false, "audio": false },
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

/// `/props`' `engine` object in toktape's shape: the server's own `name`,
/// `version` (with the engine's note), `args` (this process's argv, verbatim)
/// and `server_pid`, and the model, placement, draft and verified context
/// the engine reports.
pub(crate) fn engine_object(p: &EngineProps) -> Value {
    let mut o = Map::new();
    o.insert("name".into(), json!("bloomery"));
    let version = match &p.version_note {
        Some(note) => format!("{VERSION} {note}"),
        None => VERSION.to_owned(),
    };
    o.insert("version".into(), Value::String(version));
    let args = std::env::args_os()
        .map(|a| Value::String(a.to_string_lossy().into_owned()))
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
        let rows: [(&str, &str, &str, String); 17] = [
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
        let mut out = String::new();
        for (kind, name, help, value) in rows {
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
    Ok(json!({ "prompt": render_chat(state, b)? }))
}

/// Renders the chat template for an OpenAI `messages` body.
fn render_chat(state: &State, b: &Map<String, Value>) -> Result<String, ApiError> {
    let Some(Value::Array(msgs)) = b.get("messages") else {
        return Err(invalid("'messages' is required and must be an array"));
    };
    let mut messages = Vec::with_capacity(msgs.len());
    for m in msgs {
        let Value::Object(m) = m else {
            return Err(invalid("each message must be an object"));
        };
        if !m.get("role").is_some_and(Value::is_string) {
            return Err(invalid("each message needs a string 'role'"));
        }
        let mut m = m.clone();
        if let Some(Value::Array(parts)) = m.get("content") {
            let text: String = parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            m.insert("content".into(), Value::String(text));
        }
        messages.push(Value::Object(m));
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
    for key in ["tools", "reasoning_effort"] {
        if let Some(v) = b.get(key).filter(|v| !v.is_null()) {
            vars.insert(key.into(), v.clone());
        }
    }
    if let Some(Value::Object(kw)) = b.get("chat_template_kwargs") {
        vars.extend(kw.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    state.template.render(&vars).map_err(|e| ApiError {
        retry_after: false,
        code: 500,
        kind: "server_error",
        message: e.to_string(),
    })
}

// ---------------------------------------------------------------- generation endpoints

fn dead_engine(reason: &str) -> ApiError {
    ApiError {
        retry_after: false,
        code: 503,
        kind: "unavailable_error",
        message: format!("the engine failed and the server is stopping: {reason}"),
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
/// channel without a last message: the failure that ended it, a 500 for a
/// request that had started, else a 503.
fn engine_gone(state: &State, started: bool) -> ApiError {
    let reason = state
        .fatal()
        .unwrap_or_else(|| "the engine thread ended".to_owned());
    if started {
        engine_error(&reason)
    } else {
        dead_engine(&reason)
    }
}

/// Validates the prompt, hands the request to the engine thread, and passes
/// its events to `sink` with the slot it took. Returns the outcome and the
/// slot; an error before any event is the request's answer. A sink that fails
/// ends the request: the engine thread sees its channel closed.
fn run_gen(
    state: &State,
    ids: &[u32],
    prompt: Value,
    p: &GenParams,
    sink: &mut dyn FnMut(Event<'_>, usize) -> io::Result<()>,
) -> Result<(Result<Outcome, GenError>, usize), ApiError> {
    if let Some(reason) = state.fatal() {
        return Err(dead_engine(&reason));
    }
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
        events,
    };
    relock(&state.shared.board)
        .enqueue(ids.to_vec(), submit)
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
    run: impl FnOnce(&mut Slot) -> (Result<Value, ApiError>, Option<EngineError>) + Send + 'static,
) -> Result<Value, ApiError> {
    if let Some(reason) = state.fatal() {
        return Err(dead_engine(&reason));
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
    let r = on_engine(state, Reserve::All, who, |slot| {
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
    let r = on_engine(
        state,
        Reserve::One(id),
        &format!("slot {id}"),
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
/// mixed array whose strings are tokenized in place.
fn completion_prompt(state: &State, v: Option<&Value>) -> Result<Vec<u32>, ApiError> {
    let e = &*state.tok;
    match v {
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
                    other => ids.extend(token_ids(state, &Value::Array(vec![other.clone()]))?),
                }
            }
            Ok(ids)
        }
        _ => Err(invalid("'prompt' must be a string or an array of tokens")),
    }
}

/// The last `/completion` object. `tokens` (the generated ids, the
/// end-of-generation one included) is llama.cpp's `return_tokens` field.
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

fn completion(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = body(req).and_then(|b| gen_params(state, &b).map(|p| (b, p)));
    let (b, p) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let ids = match completion_prompt(state, b.get("prompt")) {
        Ok(ids) => ids,
        Err(e) => return send_error(w, req, &e),
    };
    let return_tokens = get_b(&b, "return_tokens").unwrap_or(false);
    let prompt = b.get("prompt").cloned().unwrap_or(Value::Null);
    if !p.stream {
        return match run_gen(state, &ids, prompt.clone(), &p, &mut |_, _| Ok(())) {
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
                    v
                }
            };
            sse(s, &v)
        };
        run_gen(state, &ids, prompt.clone(), &p, &mut sink)
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
        ToolFormat::Unparsed => {
            return Err(ApiError {
                retry_after: false,
                code: 501,
                kind: "not_supported_error",
                message: "the chat template's tool-call markup has no parser in this server \
                          (DSML and GLM's are parsed): send the request without tools or with \
                          tool_choice \"none\""
                    .to_owned(),
            });
        }
    }))
}

/// A generation whose tool-call markup does not parse: the server's error,
/// as llama-server answers output its chat parser refuses.
fn tool_markup_error(e: &GlmXmlError) -> ApiError {
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

fn chat(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = body(req).and_then(|b| {
        let p = gen_params(state, &b)?;
        let format = ReasoningFormat::from_request(b.get("reasoning_format")).map_err(invalid)?;
        let text = render_chat(state, &b)?;
        let tools = tool_scan(state, &b)?;
        Ok((b, p, format, text, tools))
    });
    let (b, p, format, text, tools) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let ids_meta = ChatIds {
        id: format!("chatcmpl-{}", state.random_id()),
        created: unix_now(),
        model: b
            .get("model")
            .and_then(Value::as_str)
            .map_or_else(|| state.alias.clone(), str::to_owned),
    };
    let ids = state.tok.encode(&text);
    let mut parser = ChatParser::with_tools(&text, format, tools);
    let prompt = Value::String(text);
    if !p.stream {
        return match run_gen(state, &ids, prompt, &p, &mut |_, _| Ok(())) {
            Err(e) => send_error(w, req, &e),
            Ok((Err(e), _)) => send_error(w, req, &engine_error(&e)),
            Ok((Ok(o), _)) => match parser
                .try_push(&o.content)
                .and_then(|_| parser.try_finish())
            {
                Ok(_) => send_json(w, req, 200, &chat_final(&ids_meta, &o, parser.message())),
                Err(e) => send_error(w, req, &tool_markup_error(&e)),
            },
        };
    }
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    // Markup that does not parse stops the generation (the sink fails) and
    // ends the stream with an error event instead of a dropped connection.
    let mut markup: Option<GlmXmlError> = None;
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
                    for delta in meta.deltas(&d) {
                        let mut v = meta.chunk(json!([{
                            "finish_reason": null, "index": 0, "delta": delta,
                        }]));
                        if tpt {
                            v["timings"] = t.to_json();
                        }
                        sse(s, &v)?;
                    }
                    Ok(())
                }
            }
        };
        run_gen(state, &ids, prompt, &p, &mut sink).map(|(o, _)| o)
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
        for delta in ids_meta.deltas(&d) {
            sse(
                s,
                &ids_meta.chunk(json!([{ "finish_reason": null, "index": 0, "delta": delta }])),
            )?;
        }
        let mut last = ids_meta.chunk(json!([{
            "finish_reason": chat_finish_reason(o, parser.message()), "index": 0, "delta": {},
        }]));
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
/// non-empty, as llama-server writes them.
fn chat_final(meta: &ChatIds, o: &Outcome, m: &Message) -> Value {
    let mut message = json!({ "role": "assistant", "content": m.content });
    if !m.reasoning.is_empty() {
        message["reasoning_content"] = json!(m.reasoning);
    }
    if !m.calls.is_empty() {
        message["tool_calls"] = m.calls.iter().map(|c| meta.tool_call(c)).collect();
    }
    json!({
        "choices": [{
            "finish_reason": chat_finish_reason(o, m),
            "index": 0,
            "message": message,
        }],
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
    use super::{
        ACCEPT_BACKOFF, EBADF, EFAULT, EINVAL, EMFILE, ENFILE, EngineProps, SlotQueue,
        after_accept_error, engine_object,
    };
    use std::io;
    use std::time::Duration;

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
}
