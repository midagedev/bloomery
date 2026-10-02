//! Gate: N slots (`--parallel N`) over the mock engine of N slots — each slot's
//! ids are its request's alone, a free slot takes the oldest waiting request,
//! a request takes the slot whose ids share its prefix, the queue refuses past
//! its depth, `/slots` serves N ids and no more, and N past what the engine
//! declares is refused by name.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use serve::{
    Engine, EngineError, FATAL_LINGER, MockEngine, ServeError, Server, ServerConfig, SlotConfig,
    SlotRow, Tokenizer,
};

use super::common::{V41_TEMPLATE, call, get, post};
use super::{EntryLog, assert_error, metric, wait_deferred};

const BOUND: Duration = Duration::from_secs(10);

fn config(dir: Option<&Path>) -> ServerConfig {
    ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: V41_TEMPLATE.to_owned(),
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: dir.map(Path::to_path_buf),
    }
}

/// A server of `n` slots, the queue `depth` deep (`None`: the default).
fn start_n(
    engine: Box<dyn Engine>,
    n: usize,
    depth: Option<usize>,
    dir: Option<&Path>,
) -> SocketAddr {
    let slots = SlotConfig {
        parallel: n,
        queue_depth: depth,
        ..SlotConfig::default()
    };
    let server = Server::bind_with("127.0.0.1:0", engine, config(dir), slots)
        .unwrap_or_else(|e| panic!("bind: {e}"));
    server.spawn().expect("spawn")
}

/// The mock of `slots` slots with an [`EntryLog`] in `prefill` (every entry
/// recorded and, until the log is released, held there) and the width of
/// every step of several slots recorded.
struct Gated {
    inner: MockEngine,
    log: Arc<EntryLog>,
    widths: Arc<Mutex<Vec<usize>>>,
}

impl Gated {
    fn new(slots: usize, log: &Arc<EntryLog>, widths: &Arc<Mutex<Vec<usize>>>) -> Gated {
        Gated {
            inner: MockEngine::new(4096).with_slots(slots),
            log: Arc::clone(log),
            widths: Arc::clone(widths),
        }
    }
}

impl Engine for Gated {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.log.enter(self.inner.tokenizer().decode(ids));
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.inner.next(last, out)
    }
    fn reset(&mut self) -> Result<(), EngineError> {
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
        self.inner.describe()
    }
    fn slots(&self) -> usize {
        self.inner.slots()
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.inner.select_slot(slot)
    }
    fn step_slots(&mut self, rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
        self.widths.lock().expect("widths").push(rows.len());
        self.inner.step_slots(rows)
    }
}

/// Prompts whose greedy ids change when another prompt's ids share their
/// context: each reuses the others' letters in another order.
const PROMPTS: [&str; 4] = ["abcabcab", "acbacbac", "bcabcabc", "cbacbacb"];

fn completion(prompt: &str, n_predict: usize) -> Value {
    json!({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "return_tokens": true})
}

/// At N = 2 and N = 4 the requests run together — every step after the
/// prompts carries all N slots — and each slot's greedy ids are its request's
/// run alone on a one-slot server.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_each_slot_gives_its_request_alone_ids() {
    let alone = super::common::start(4096);
    for n in [2, 4] {
        let log = Arc::new(EntryLog::default());
        let widths = Arc::new(Mutex::new(Vec::new()));
        let addr = start_n(Box::new(Gated::new(n, &log, &widths)), n, None, None);
        let mut workers = Vec::new();
        for (i, p) in PROMPTS[..n].iter().enumerate() {
            let body = completion(p, 12);
            workers.push(std::thread::spawn(move || post(addr, "/completion", &body)));
            if i == 0 {
                assert!(
                    log.len_reached(1, BOUND),
                    "the first prompt never reached the engine"
                );
            } else {
                wait_deferred(addr, i, BOUND);
            }
        }
        log.release();
        let mut slots_seen = Vec::new();
        for (p, w) in PROMPTS[..n].iter().zip(workers) {
            let r = w.join().expect("worker");
            assert_eq!(r.status, 200, "{}", r.body);
            let v = r.json();
            let want = post(alone, "/completion", &completion(p, 12)).json();
            assert_eq!(v["tokens"], want["tokens"], "N = {n}, {p:?}: {v}");
            assert_eq!(v["content"], want["content"], "N = {n}, {p:?}");
            slots_seen.push(v["id_slot"].as_u64().expect("id_slot"));
        }
        slots_seen.sort_unstable();
        let every: Vec<u64> = (0..n as u64).collect();
        assert_eq!(slots_seen, every, "N = {n}: each request its own slot");
        let widths = widths.lock().expect("widths").clone();
        assert_eq!(
            widths.iter().max(),
            Some(&n),
            "N = {n}: no step carried every slot: {widths:?}"
        );
        let per = metric(&get(addr, "/metrics").body, "n_busy_slots_per_decode");
        assert!(per > 1.0, "N = {n}: {per} busy slots per engine call");
    }
}

/// At N = 2 the requests queued behind a held engine get slots in arrival
/// order: a free slot takes the oldest waiting request.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_free_slot_takes_the_oldest_waiting_request() {
    const QUEUED: usize = 4;
    let log = Arc::new(EntryLog::default());
    let widths = Arc::new(Mutex::new(Vec::new()));
    let addr = start_n(Box::new(Gated::new(2, &log, &widths)), 2, None, None);
    let mut workers = Vec::new();
    for i in 0..=QUEUED {
        let body = json!({"prompt": format!("{i}a"), "n_predict": 3, "temperature": 0});
        workers.push(std::thread::spawn(move || post(addr, "/completion", &body)));
        if i == 0 {
            assert!(
                log.len_reached(1, BOUND),
                "the first prompt never reached the engine"
            );
        } else {
            wait_deferred(addr, i, BOUND);
        }
    }
    log.release();
    for w in workers {
        let r = w.join().expect("worker");
        assert_eq!(r.status, 200, "{}", r.body);
    }
    let want: Vec<String> = (0..=QUEUED).map(|i| i.to_string()).collect();
    assert_eq!(
        log.prompts(),
        want,
        "the slots must take requests in arrival order"
    );
}

/// Of two free slots a request takes the one whose ids share its prefix, not
/// the least recently used one; sharing none it takes the least recently used.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_request_takes_the_slot_that_shares_its_prefix() {
    let addr = start_n(Box::new(MockEngine::new(4096).with_slots(2)), 2, None, None);
    let slot_of = |prompt: &str| {
        let v = post(addr, "/completion", &completion(prompt, 4)).json();
        (
            v["id_slot"].as_u64().expect("id_slot"),
            v["timings"]["cache_n"].clone(),
        )
    };
    assert_eq!(slot_of("abcabcabc").0, 0, "both unused: the lowest id");
    assert_eq!(
        slot_of("xyzxyzxyz").0,
        1,
        "nothing shared: the least recently used"
    );
    let (slot, kept) = slot_of("xyzxyzxyzxyzq");
    assert_eq!(
        slot, 1,
        "slot 1 holds the prefix, slot 0 is the least recently used"
    );
    assert_eq!(kept, 12, "the slot's twelve held ids");
    assert_eq!(
        slot_of("qqqqq").0,
        0,
        "nothing shared: the least recently used"
    );
}

/// Past the queue's depth a request is a 503 with `Retry-After` at once, never
/// queued; the requests in the slot and the queue are served.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_request_past_the_queue_depth_is_503() {
    let log = Arc::new(EntryLog::default());
    let widths = Arc::new(Mutex::new(Vec::new()));
    let addr = start_n(Box::new(Gated::new(1, &log, &widths)), 1, Some(1), None);
    let body = json!({"prompt": "abcab", "n_predict": 2, "temperature": 0});
    let first = {
        let body = body.clone();
        std::thread::spawn(move || post(addr, "/completion", &body))
    };
    assert!(
        log.len_reached(1, BOUND),
        "the first prompt never reached the engine"
    );
    let queued = {
        let body = body.clone();
        std::thread::spawn(move || post(addr, "/completion", &body))
    };
    wait_deferred(addr, 1, BOUND);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(post(addr, "/completion", &body));
    });
    let past = rx.recv_timeout(BOUND);
    log.release();
    let past = past.expect("the request past the depth waited instead of a 503");
    assert_error(&past, 503, "unavailable_error", "the queue's depth");
    assert_eq!(past.header("Retry-After"), Some("1"), "{:?}", past.headers);
    assert_eq!(first.join().expect("first").status, 200);
    assert_eq!(queued.join().expect("queued").status, 200);
    assert_eq!(
        metric(&get(addr, "/metrics").body, "requests_deferred"),
        0.0
    );
}

/// `/slots` lists N slots and `/props` reports N; a slot action takes any id
/// below N, answers 503 on a busy slot, and an id of N or more is a 400 naming
/// it.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slot_ids_past_the_slots_are_400() {
    let dir = super::common::fresh_dir("slots-n");
    let latch = Arc::new(super::Latch::default());
    let held = super::Held::new(4096, 3, Arc::clone(&latch));
    let addr = start_n(Box::new(Slotted { held, slots: 2 }), 2, None, Some(&dir));
    assert_eq!(get(addr, "/props").json()["total_slots"], 2);
    let list = get(addr, "/slots").json();
    let ids: Vec<Value> = list
        .as_array()
        .expect("a list")
        .iter()
        .map(|s| s["id"].clone())
        .collect();
    assert_eq!(ids, [json!(0), json!(1)]);
    let gen_thread =
        std::thread::spawn(move || post(addr, "/completion", &completion("abcabc", 5)));
    assert!(
        latch.wait_entered(BOUND),
        "the generation never reached next #3"
    );
    let busy = call(addr, "POST", "/slots/0?action=erase", None);
    let past = call(addr, "POST", "/slots/2?action=erase", None);
    let health = get(addr, "/health").json();
    latch.release();
    assert_error(
        &busy,
        503,
        "unavailable_error",
        "slot 0 is processing a request",
    );
    assert_error(&past, 400, "invalid_request_error", "Invalid slot ID 2");
    assert_eq!(health["slots_processing"], 1, "{health}");
    assert_eq!(health["slots_idle"], 1, "{health}");
    assert_eq!(gen_thread.join().expect("generation").status, 200);
    let one = call(addr, "POST", "/slots/1?action=erase", None);
    assert_eq!(one.status, 200, "{}", one.body);
    assert_eq!(one.json(), json!({"id_slot": 1, "n_erased": 0}));
    let zero = call(addr, "POST", "/slots/0?action=erase", None);
    assert_eq!(zero.json(), json!({"id_slot": 0, "n_erased": 10}));
    super::common::drop_dir(&dir);
}

/// [`super::Held`] declaring `slots` slots: one context, which this gate's
/// single request on slot 0 is all that uses.
struct Slotted {
    held: super::Held,
    slots: usize,
}

impl Engine for Slotted {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.held.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.held.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.held.next(last, out)
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.held.reset()
    }
    fn ctx_max(&self) -> usize {
        self.held.ctx_max()
    }
    fn describe(&self) -> String {
        self.held.describe()
    }
    fn slots(&self) -> usize {
        self.slots
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        if slot < self.slots {
            Ok(())
        } else {
            Err(EngineError(format!("slot {slot}")))
        }
    }
}

/// The mock that drafts, declaring two slots: a drafting engine is refused
/// past one slot whatever it declares.
struct DraftingSlots(serve::DraftMock);

impl Engine for DraftingSlots {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.0.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.0.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.0.next(last, out)
    }
    fn advance_rows(&self) -> usize {
        self.0.advance_rows()
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.0.reset()
    }
    fn ctx_max(&self) -> usize {
        self.0.ctx_max()
    }
    fn describe(&self) -> String {
        self.0.describe()
    }
    fn slots(&self) -> usize {
        2
    }
}

/// N past the slots an engine declares is refused by name before the server
/// binds, as are N = 0, N on an engine that drafts, and a depth no queue can
/// reach; the mock binary exits 64 on each.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slots_an_engine_does_not_declare_are_refused() {
    let refusal = |engine: Box<dyn Engine>, parallel: usize, queue_depth: Option<usize>| {
        let slots = SlotConfig {
            parallel,
            queue_depth,
            ..SlotConfig::default()
        };
        match Server::bind_with("127.0.0.1:0", engine, config(None), slots) {
            Err(ServeError::Slots(m)) => m,
            Err(e) => panic!("--parallel {parallel}: another error: {e}"),
            Ok(_) => panic!("--parallel {parallel} queue {queue_depth:?}: bound"),
        }
    };
    let m = refusal(Box::new(MockEngine::new(64)), 2, None);
    assert!(
        m.contains("--parallel 2") && m.contains("serves 1 slot"),
        "{m}"
    );
    let m = refusal(Box::new(MockEngine::new(64).with_slots(2)), 3, None);
    assert!(
        m.contains("--parallel 3") && m.contains("serves 2 slot"),
        "{m}"
    );
    let m = refusal(Box::new(MockEngine::new(64)), 0, None);
    assert!(m.contains("--parallel 0"), "{m}");
    let m = refusal(Box::new(DraftingSlots(serve::DraftMock::new(64))), 2, None);
    assert!(m.contains("drafts"), "{m}");
    let most = serve::MAX_CONNECTIONS - 2;
    let m = refusal(
        Box::new(MockEngine::new(64).with_slots(2)),
        2,
        Some(most + 1),
    );
    assert!(m.contains(&format!("--queue-depth {}", most + 1)), "{m}");
    for args in [["--parallel", "0"], ["-np", "65"]] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_bloomery-serve"))
            .args(["--port", "0"])
            .args(args)
            .output()
            .expect("run bloomery-serve");
        assert_eq!(out.status.code(), Some(64), "{args:?}: {out:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("--parallel"), "{args:?}: {err}");
    }
}
