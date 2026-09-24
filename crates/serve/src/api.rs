//! Routes and JSON shapes. The field names, defaults and stream framing follow
//! llama-server (ik_llama.cpp `examples/server`) so its clients work unchanged.
//!
//! One slot: the engine sits behind a mutex and a second generation waits for
//! the first. `/health`, `/props`, `/slots`, `/metrics`, `/v1/models` and the
//! tokenizer endpoints never take that mutex.
//!
//! An engine error is fatal: the request that met it gets a 500 carrying the
//! engine's message, every later generation and `/health` a 503 with the reason,
//! and after [`ServerConfig::fatal_linger`] [`Server::run`] returns
//! [`ServeError::Engine`] so the process exits instead of holding the port with a
//! dead engine behind it.

use std::fmt;
use std::io::{self, BufReader};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::dsml::{ChatParser, Message, ToolCall};
use crate::engine::{Engine, SamplerFactory, SamplingParams, Tokenizer};
use crate::genloop::{self, Event, GenError, GenParams, Outcome, Slot, Timings};
use crate::http::{self, EventStream, Request};
use crate::reasoning::ReasoningFormat;
use crate::sampling;
use crate::template::{ChatTemplate, TemplateError};

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
}

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
    #[error("bind: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Template(#[from] TemplateError),
    #[error("{0}")]
    Engine(EngineFailure),
}

/// Why the accept loop ended.
enum End {
    Io(io::Error),
    Engine(EngineFailure),
}

/// A bound, not yet running server.
pub struct Server {
    listener: TcpListener,
    state: Arc<State>,
    ended: mpsc::Receiver<End>,
}

impl Server {
    /// Binds `addr` and takes ownership of the engine.
    pub fn bind(
        addr: impl ToSocketAddrs,
        engine: Box<dyn Engine>,
        config: ServerConfig,
    ) -> Result<Self, ServeError> {
        let template = ChatTemplate::parse(&config.chat_template)?;
        let listener = TcpListener::bind(addr)?;
        let tok = engine.tokenizer();
        let info = ModelInfo {
            n_vocab: tok.n_vocab(),
            ctx_max: engine.ctx_max(),
            bos_text: tok.decode(&[tok.bos()]),
            eos_text: tok.decode(&[tok.eos()]),
        };
        let (end_tx, ended) = mpsc::channel();
        let state = State {
            slot_engine: Mutex::new(Slot::new(engine)),
            tok,
            fatal: Mutex::new(None),
            end: end_tx,
            fatal_linger: config.fatal_linger,
            template,
            alias: config.model_alias,
            model_path: config.model_path,
            sampler: config.sampler.unwrap_or_else(sampling::reference_factory),
            info,
            busy: AtomicBool::new(false),
            waiting: AtomicUsize::new(0),
            stats: Mutex::new(Stats::default()),
            slot: Mutex::new(SlotView::default()),
            start_unix: unix_now(),
            ids: AtomicU64::new(0),
        };
        Ok(Server {
            listener,
            state: Arc::new(state),
            ended,
        })
    }

    /// The bound address (the real port when bound to port 0).
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accepts connections, one thread each, until the listener fails or the
    /// engine does, and returns why.
    pub fn run(self) -> ServeError {
        let Server {
            listener,
            state,
            ended,
        } = self;
        let accept_state = Arc::clone(&state);
        thread::spawn(move || {
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => {
                        let state = Arc::clone(&accept_state);
                        thread::spawn(move || serve_conn(&state, stream));
                    }
                    Err(e) => {
                        let _ = accept_state.end.send(End::Io(e));
                        return;
                    }
                }
            }
        });
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

    /// Runs the accept loop on a background thread and returns the address.
    pub fn spawn(self) -> io::Result<SocketAddr> {
        let addr = self.local_addr()?;
        thread::spawn(move || self.run());
        Ok(addr)
    }
}

struct ModelInfo {
    n_vocab: usize,
    ctx_max: usize,
    bos_text: String,
    eos_text: String,
}

#[derive(Default)]
struct Stats {
    n_prompt_total: u64,
    t_prompt_ms_total: f64,
    n_predicted_total: u64,
    t_predicted_ms_total: f64,
    n_decode_total: u64,
    n_busy_slots_total: u64,
}

#[derive(Default)]
struct SlotView {
    id_task: u64,
    prompt: Value,
    settings: Value,
    /// The running request's `n_predict` (`-1` unbounded).
    n_predict: i64,
    n_past: usize,
    n_decoded: usize,
    stopped_eos: bool,
    stopped_word: bool,
    stopped_limit: bool,
    stopping_word: String,
}

struct State {
    /// The engine and the ids its cache holds, under one lock.
    slot_engine: Mutex<Slot>,
    /// The engine's vocabulary, read without the engine lock.
    tok: Arc<dyn Tokenizer>,
    /// Set by the request that met an engine error; the reason `/health` gives.
    fatal: Mutex<Option<String>>,
    end: mpsc::Sender<End>,
    fatal_linger: Duration,
    template: ChatTemplate,
    alias: String,
    model_path: String,
    sampler: SamplerFactory,
    info: ModelInfo,
    busy: AtomicBool,
    waiting: AtomicUsize,
    stats: Mutex<Stats>,
    slot: Mutex<SlotView>,
    start_unix: u64,
    ids: AtomicU64,
}

fn relock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl State {
    /// Waits for the one slot (counted in `requests_deferred` while waiting).
    fn engine(&self) -> MutexGuard<'_, Slot> {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let g = relock(&self.slot_engine);
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        g
    }

    fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::SeqCst)
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

fn serve_conn(state: &State, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(600)));
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::new(stream);
    loop {
        let req = match http::read_request(&mut r, &mut w) {
            Ok(Some(req)) => req,
            Ok(None) => return,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                let body = error_body(400, "invalid_request_error", &e.to_string());
                let req = Request {
                    method: String::new(),
                    path: String::new(),
                    query: String::new(),
                    http10: false,
                    headers: Vec::new(),
                    body: Vec::new(),
                    keep_alive: false,
                };
                let _ = http::respond(&mut w, &req, 400, JSON, &[], body.to_string().as_bytes());
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

const JSON: &str = "application/json; charset=utf-8";

/// An error that becomes an OpenAI-style error object.
struct ApiError {
    code: u16,
    kind: &'static str,
    message: String,
}

fn invalid(message: impl Into<String>) -> ApiError {
    ApiError {
        code: 400,
        kind: "invalid_request_error",
        message: message.into(),
    }
}

fn error_body(code: u16, kind: &str, message: &str) -> Value {
    json!({ "error": { "code": code, "message": message, "type": kind } })
}

fn send_json(w: &mut TcpStream, req: &Request, status: u16, v: &Value) -> io::Result<bool> {
    http::respond(w, req, status, JSON, &[], v.to_string().as_bytes())?;
    Ok(true)
}

fn send_error(w: &mut TcpStream, req: &Request, e: &ApiError) -> io::Result<bool> {
    send_json(w, req, e.code, &error_body(e.code, e.kind, &e.message))
}

/// Dispatches one request; `Ok(true)` when the connection may carry another.
fn route(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let r = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health" | "/v1/health") => return health(state, req, w),
        ("GET", "/v1/models" | "/models") => Ok(models(state)),
        ("GET", "/props") => Ok(props(state)),
        ("GET", "/slots") => Ok(slots(state)),
        ("GET", "/metrics") => return metrics(state, req, w),
        ("POST", "/completion" | "/completions") => return completion(state, req, w),
        ("POST", "/v1/chat/completions" | "/chat/completions") => return chat(state, req, w),
        ("POST", "/tokenize") => body(req).and_then(|b| tokenize(state, &b)),
        ("POST", "/detokenize") => body(req).and_then(|b| detokenize(state, &b)),
        ("POST", "/apply-template") => body(req).and_then(|b| apply_template(state, &b)),
        ("OPTIONS", _) => {
            http::respond(
                w,
                req,
                204,
                "text/plain",
                &[
                    (
                        "Access-Control-Allow-Methods",
                        "GET, POST, OPTIONS".to_owned(),
                    ),
                    ("Access-Control-Allow-Headers", "*".to_owned()),
                ],
                b"",
            )?;
            return Ok(true);
        }
        _ => Err(ApiError {
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
    Ok(GenParams {
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
    })
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
    if let Some(reason) = relock(&state.fatal).clone() {
        let mut v = error_body(503, "unavailable_error", &reason);
        v["status"] = json!("error");
        v["reason"] = json!(reason);
        return send_json(w, req, 503, &v);
    }
    let busy = state.busy.load(Ordering::SeqCst);
    let v = json!({
        "status": if busy { "no slot available" } else { "ok" },
        "slots_idle": i32::from(!busy),
        "slots_processing": i32::from(busy),
    });
    let status = if busy && req.has_query("fail_on_no_slot") {
        503
    } else {
        200
    };
    send_json(w, req, status, &v)
}

fn models(state: &State) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": state.alias,
            "object": "model",
            "created": state.start_unix,
            "owned_by": "bloomery",
            "meta": { "n_vocab": state.info.n_vocab, "n_ctx_train": state.info.ctx_max },
            "max_model_len": state.info.ctx_max,
        }],
    })
}

fn props(state: &State) -> Value {
    json!({
        "system_prompt": "",
        "model_alias": state.alias,
        "model_path": state.model_path,
        "model_name": state.alias,
        "default_generation_settings": generation_settings(state, &default_params()),
        "total_slots": 1,
        "chat_template": state.template.source(),
        "chat_template_caps": {},
        "bos_token": state.info.bos_text,
        "eos_token": state.info.eos_text,
        "modalities": { "vision": false, "audio": false },
        "n_ctx": state.info.ctx_max,
        "build_info": concat!("bloomery-serve ", env!("CARGO_PKG_VERSION")),
    })
}

fn slots(state: &State) -> Value {
    let busy = state.busy.load(Ordering::SeqCst);
    let s = relock(&state.slot);
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
        m.insert("id".into(), json!(0));
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
    Value::Array(vec![v])
}

fn metrics(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let busy = state.busy.load(Ordering::SeqCst);
    let deferred = state.waiting.load(Ordering::SeqCst);
    let n_past = relock(&state.slot).n_past;
    let kv_ratio = if state.info.ctx_max > 0 {
        n_past as f64 / state.info.ctx_max as f64
    } else {
        0.0
    };
    let text = {
        let s = relock(&state.stats);
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
        let rows: [(&str, &str, &str, String); 12] = [
            (
                "counter",
                "prompt_tokens_total",
                "Number of prompt tokens processed.",
                s.n_prompt_total.to_string(),
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
                u8::from(busy).to_string(),
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
        code: 500,
        kind: "server_error",
        message: e.to_string(),
    })
}

// ---------------------------------------------------------------- generation endpoints

/// Holds the slot for one generation: busy flag, slot view, counters. Dropped
/// after the response is written; an engine failure it met is signalled then.
struct Run<'a> {
    state: &'a State,
    engine: MutexGuard<'a, Slot>,
    failure: Option<EngineFailure>,
}

fn dead_engine(reason: &str) -> ApiError {
    ApiError {
        code: 503,
        kind: "unavailable_error",
        message: format!("the engine failed and the server is stopping: {reason}"),
    }
}

impl<'a> Run<'a> {
    /// Waits for the slot; refuses once the engine has failed.
    fn begin(state: &'a State) -> Result<Self, ApiError> {
        let engine = state.engine();
        if let Some(reason) = relock(&state.fatal).as_deref() {
            return Err(dead_engine(reason));
        }
        state.busy.store(true, Ordering::SeqCst);
        Ok(Run {
            state,
            engine,
            failure: None,
        })
    }

    /// Validates the prompt and runs the loop, then books the counters.
    fn go(
        &mut self,
        ids: &[u32],
        prompt: Value,
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<Result<Outcome, GenError>, ApiError> {
        if ids.is_empty() {
            return Err(invalid("the prompt is empty"));
        }
        if ids.len() >= self.state.info.ctx_max {
            return Err(ApiError {
                code: 400,
                kind: "exceed_context_size_error",
                message: format!(
                    "the prompt has {} tokens and the context holds {}",
                    ids.len(),
                    self.state.info.ctx_max
                ),
            });
        }
        {
            let mut s = relock(&self.state.slot);
            *s = SlotView {
                id_task: self.state.next_id(),
                prompt,
                settings: generation_settings(self.state, p),
                n_predict: p.n_predict,
                ..SlotView::default()
            };
        }
        let state = self.state;
        let mut tick = |t: &Timings| {
            let mut v = relock(&state.slot);
            v.n_decoded = t.predicted_n;
            v.n_past = t.n_past;
        };
        let mut tim = Timings::default();
        let r = genloop::generate(
            &mut self.engine,
            &self.state.sampler,
            ids,
            p,
            sink,
            &mut tick,
            &mut tim,
        );
        self.book(&tim, r.as_ref().ok());
        if let Err(GenError::Engine(e)) = &r {
            let f = EngineFailure {
                engine: self.engine.engine.describe(),
                error: e.to_string(),
            };
            *relock(&self.state.fatal) = Some(f.error.clone());
            self.failure = Some(f);
        }
        Ok(r)
    }

    fn book(&self, t: &Timings, o: Option<&Outcome>) {
        let mut s = relock(&self.state.stats);
        let (pn, dn) = (t.prompt_n as u64, t.predicted_n as u64);
        s.n_prompt_total += pn;
        s.t_prompt_ms_total += t.prompt_ms;
        s.n_predicted_total += dn;
        s.t_predicted_ms_total += t.predicted_ms;
        // llama_decode calls: one for the prompt, whose logits give the first
        // generated token, then one per later token (the last is never fed back).
        let decodes = match (pn, dn) {
            (0, _) => 0,
            (_, 0) => 1,
            (_, d) => d,
        };
        s.n_decode_total += decodes;
        s.n_busy_slots_total += decodes;
        drop(s);
        let mut v = relock(&self.state.slot);
        v.n_past = t.n_past;
        v.n_decoded = t.predicted_n;
        if let Some(o) = o {
            v.stopped_eos = o.stop == genloop::StopKind::Eos;
            v.stopped_word = o.stop == genloop::StopKind::Word;
            v.stopped_limit = o.stop == genloop::StopKind::Limit;
            v.stopping_word = o.stopping_word.clone();
        }
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        self.state.busy.store(false, Ordering::SeqCst);
        if let Some(f) = self.failure.take() {
            let _ = self.state.end.send(End::Engine(f));
        }
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
) -> Value {
    let mut v = json!({
        "content": if p.stream { "" } else { o.content.as_str() },
        "generated_text": o.content,
        "id_slot": 0,
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

fn engine_error(e: &GenError) -> ApiError {
    ApiError {
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
    let mut run = match Run::begin(state) {
        Ok(r) => r,
        Err(e) => return send_error(w, req, &e),
    };
    let prompt = b.get("prompt").cloned().unwrap_or(Value::Null);
    if !p.stream {
        return match run.go(&ids, prompt.clone(), &p, &mut |_| Ok(())) {
            Err(e) => send_error(w, req, &e),
            Ok(Err(e)) => send_error(w, req, &engine_error(&e)),
            Ok(Ok(o)) => send_json(
                w,
                req,
                200,
                &completion_final(state, &o, &p, &prompt, return_tokens),
            ),
        };
    }
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    let r = {
        let mut sink = |ev: Event<'_>| -> io::Result<()> {
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
                    "content": "", "stop": false, "id_slot": 0, "multimodal": false,
                    "prompt_progress": progress(t),
                }),
                Event::Text(text, t) => {
                    let mut v = json!({ "content": text, "stop": false, "id_slot": 0, "multimodal": false });
                    if tpt {
                        v["timings"] = t.to_json();
                    }
                    v
                }
            };
            sse(s, &v)
        };
        run.go(&ids, prompt.clone(), &p, &mut sink)
    };
    finish_stream(req, stream, w_opt, r, |s, o| {
        sse(s, &completion_final(state, o, &p, &prompt, return_tokens))
    })
}

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

/// Whether the output is scanned for DSML: non-empty `tools` and a `tool_choice`
/// other than `"none"` (the template still sees the tools with `"none"`).
fn parses_tools(b: &Map<String, Value>) -> bool {
    b.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
        && b.get("tool_choice").and_then(Value::as_str) != Some("none")
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
        Ok((b, p, format, text))
    });
    let (b, p, format, text) = match parsed {
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
    let mut parser = ChatParser::new(&text, format, parses_tools(&b));
    let mut run = match Run::begin(state) {
        Ok(r) => r,
        Err(e) => return send_error(w, req, &e),
    };
    let prompt = Value::String(text);
    if !p.stream {
        return match run.go(&ids, prompt, &p, &mut |_| Ok(())) {
            Err(e) => send_error(w, req, &e),
            Ok(Err(e)) => send_error(w, req, &engine_error(&e)),
            Ok(Ok(o)) => {
                let _ = parser.push(&o.content);
                let _ = parser.finish();
                send_json(w, req, 200, &chat_final(&ids_meta, &o, parser.message()))
            }
        };
    }
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    let r = {
        let meta = &ids_meta;
        let mut sink = |ev: Event<'_>| -> io::Result<()> {
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
                    for delta in meta.deltas(&parser.push(text)) {
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
        run.go(&ids, prompt, &p, &mut sink)
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
        for delta in ids_meta.deltas(&parser.finish()) {
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
