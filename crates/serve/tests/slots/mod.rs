//! Gate: N slots (`--parallel N`) over the mock engine of N slots — each slot's
//! ids are its request's alone, a free slot takes the oldest waiting request,
//! a request takes the slot whose ids share its prefix, and the conversation
//! that slot held goes to the prompt cache before the cut (on one slot, only a
//! cut to less than half), the queue refuses past its depth, `/slots` serves N
//! ids and no more, and N past what the engine declares is refused by name.
//! Then the drafting mock of N slots whose draft state is per slot: two
//! drafted streams interleave by passes, a plain step row beside a pass row,
//! and a drafting engine without the per-slot declaration is refused by
//! name. Then N slots that take the one-slot mock in
//! turns ([`serve::SwapEngine`]): preemption at a step, turns of `QUANTUM`
//! tokens, shortest prompt first, a newcomer refused by name when the running
//! request cannot be parked, the re-prefill fallback, `/slots`' turns, the
//! draft kept, a lone request on the engine's slot among equals, a slot the
//! engine emptied holding nothing for the next request, an erase of a slot
//! the engine does not hold leaving the engine's slot and moving no state,
//! a parked idle state taken into the prompt cache without a copy, and the
//! idle states the turns stop parking taken into it too. A prompt run a call a
//! round beside the decoding slots is [`yields`]'s.

mod yields;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use serve::{
    CacheNote, Drafted, Engine, EngineError, FATAL_LINGER, MockEngine, MockTokenizer, Park,
    QUANTUM, SavedState, ServeError, Server, ServerConfig, SlotConfig, SlotPass, SlotRow,
    StateError, SwapEngine, Tokenizer,
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

/// A new conversation that takes an idle slot for the header it shares cuts
/// that slot's conversation, which a server of several slots saves first: on
/// two resident slots, slot 0 holding A and slot 1 empty, D takes slot 0 for
/// the header, and A back keeps every position it held, from the prompt
/// cache. A server of one slot keeps llama-server's rule — a cut that keeps
/// half the slot or more saves nothing — and A back keeps the header alone.
/// Every request gives its alone ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_cut_of_an_idle_conversation_saves_it_on_several_slots() {
    const HEADER: &str = "system: keep it short. ";
    let (a, d) = (format!("{HEADER}abcabcab"), format!("{HEADER}xyzxyzxy"));
    let header = MockTokenizer.encode(HEADER).len();
    let alone = super::common::start(4096);
    for n in [2, 1] {
        let engine = MockEngine::new(4096).with_slots(n).with_cache_ram(1 << 20);
        let addr = start_n(Box::new(engine), n, None, None);
        let first = post(addr, "/completion", &completion(&a, 4)).json();
        let tokens = first["tokens"].as_array().expect("tokens").clone();
        let held = MockTokenizer.encode(&a).len() + tokens.len() - 1;
        assert!(
            tokens.len() == 4
                && !tokens.contains(&json!(MockTokenizer.eos()))
                && 2 * header >= held,
            "the fixture: A's four ids with no end of sequence, the header at least half of the \
             {held} positions A holds: {first}"
        );
        let other = post(addr, "/completion", &completion(&d, 4)).json();
        assert_eq!(
            (&other["id_slot"], &other["timings"]["cache_n"]),
            (&json!(0), &json!(header)),
            "N = {n}: D takes A's slot for the header: {other}"
        );
        let mut back = vec![json!(a)];
        back.extend(tokens);
        back.push(json!("zz"));
        let back = json!({"prompt": back, "n_predict": 4, "temperature": 0, "return_tokens": true});
        let v = post(addr, "/completion", &back).json();
        let want = if n > 1 { held } else { header };
        assert_eq!(
            (&v["id_slot"], &v["timings"]["cache_n"]),
            (&json!(0), &json!(want)),
            "N = {n}: A back keeps {want}: {v}"
        );
        for (got, body) in [(&other, completion(&d, 4)), (&v, back)] {
            let want = post(alone, "/completion", &body).json();
            assert_eq!(got["tokens"], want["tokens"], "N = {n}: {body}");
        }
    }
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

/// The mock that drafts, declaring two slots and no per-slot draft state: a
/// drafting engine is refused past one slot unless it declares
/// [`serve::Engine::slot_drafts`].
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

/// A drafting engine behind the entry log in `prefill` (every entry recorded
/// and, until the log is released, held there), so one request's prompt holds
/// the engine thread until the other has queued and both run together. Every
/// `select_slot`, `next` and `advance` it takes goes into `calls` in order —
/// the engine thread's single order, the deterministic record of which slot
/// each drafted pass ran on — and every `advance_slots` call with its row
/// count (`(rows, 'w')`), the record of how many drafted passes a round put
/// in one call.
struct DraftGated {
    inner: Box<dyn Engine>,
    log: Arc<EntryLog>,
    calls: Arc<Mutex<Vec<(usize, char)>>>,
}

impl Engine for DraftGated {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.log.enter(self.inner.tokenizer().decode(ids));
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.calls.lock().expect("calls").push((0, 'n'));
        self.inner.next(last, out)
    }
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        let d = self.inner.advance(last, out)?;
        self.calls.lock().expect("calls").push((0, 'a'));
        Ok(d)
    }
    fn advance_rows(&self) -> usize {
        self.inner.advance_rows()
    }
    /// The round's width into `calls`, then the rows as the default runs them
    /// — a select and an `advance` each, through this engine, so those calls
    /// are logged too.
    fn advance_slots(&mut self, rows: &mut [SlotPass<'_>]) -> Result<(), EngineError> {
        self.calls.lock().expect("calls").push((rows.len(), 'w'));
        for row in rows {
            self.select_slot(row.slot)?;
            row.drafted = self.advance(row.last, row.out)?;
        }
        Ok(())
    }
    fn slots(&self) -> usize {
        self.inner.slots()
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        let r = self.inner.select_slot(slot);
        if r.is_ok() {
            self.calls.lock().expect("calls").push((slot, 's'));
        }
        r
    }
    fn slot_drafts(&self) -> bool {
        self.inner.slot_drafts()
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
}

/// The violating engine: [`serve::DraftMock`] with its draft state — the pass
/// count that decides each pass's proposal — one count shared by both slots,
/// while it declares `slot_drafts`. The FAIL-first of
/// `hw_drafted_passes_interleave_across_slots` runs that gate's scenario
/// against it; no gate holds it, for it would pin a defect.
#[allow(dead_code, reason = "only the FAIL-first run constructs it")]
struct SharedDraft {
    inner: MockEngine,
    passes: usize,
}

#[allow(dead_code, reason = "only the FAIL-first run constructs it")]
impl SharedDraft {
    fn new(ctx_max: usize) -> Self {
        SharedDraft {
            inner: MockEngine::new(ctx_max).with_slots(2),
            passes: 0,
        }
    }
}

impl Engine for SharedDraft {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.inner.next(last, out)
    }
    /// [`serve::DraftMock`]'s rule on one counter for both slots.
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.passes += 1;
        let first = self.inner.next(last, None)?;
        let n_vocab =
            u32::try_from(serve::MockTokenizer.n_vocab()).expect("the mock's vocabulary fits u32");
        let proposal = if self.passes.is_multiple_of(3) {
            (first + 1) % n_vocab
        } else {
            first
        };
        out.push(first);
        if proposal != first {
            return Ok(Drafted {
                proposed: 1,
                accepted: 0,
            });
        }
        out.push(self.inner.next(proposal, None)?);
        Ok(Drafted {
            proposed: 1,
            accepted: 1,
        })
    }
    fn advance_rows(&self) -> usize {
        2
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
    /// The violating declaration: the draft state is shared, not per slot.
    fn slot_drafts(&self) -> bool {
        true
    }
}

/// One arrival in [`SeenChunks`]: when it landed, its stream's slot, whether
/// it is the last chunk, and the chunk itself.
type SeenChunk = (Instant, u64, bool, Value);

/// One `/completion` stream read as it arrives: every `data:` payload, with
/// the moment it landed, its stream's slot and whether it is the last chunk,
/// appended to `seen` — the shared log's order is the streams' arrival order.
fn read_stream(addr: SocketAddr, body: &Value, seen: &Mutex<Vec<SeenChunk>>) {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(60)))
        .expect("timeout");
    let text = body.to_string();
    let req = format!(
        "POST /completion HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{text}",
        text.len()
    );
    s.write_all(req.as_bytes()).expect("write");
    let mut held = String::new();
    let mut buf = [0u8; 8192];
    let mut head = true;
    loop {
        let n = match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => panic!("stream read: {e}"),
        };
        held.push_str(&String::from_utf8_lossy(&buf[..n]));
        if head {
            let Some(at) = held.find("\r\n\r\n") else {
                continue;
            };
            held.drain(..at + 4);
            head = false;
        }
        while let Some(at) = held.find("\n\n") {
            let frame = held[..at].to_owned();
            held.drain(..at + 2);
            let Some(payload) = frame.strip_prefix("data: ") else {
                panic!("a frame that is no payload: {frame}");
            };
            let v: Value = serde_json::from_str(payload).expect("chunk JSON");
            let slot = v["id_slot"].as_u64().expect("id_slot");
            let last = v["stop"].as_bool().expect("stop");
            seen.lock()
                .expect("seen")
                .push((Instant::now(), slot, last, v));
        }
    }
}

/// The slots of the drafted passes in engine-call order: each pass pairs with
/// the select that precedes it (the default `advance_slots` selects a row's
/// slot before it advances).
fn pass_slots(calls: &[(usize, char)]) -> Vec<usize> {
    let mut slot = 0;
    let mut out = Vec::new();
    for (at, what) in calls {
        match what {
            's' => slot = *at,
            'a' => out.push(slot),
            _ => {}
        }
    }
    out
}

/// Two concurrent greedy requests on the two-slot drafting mock interleave by
/// drafted passes: each request's ids and draft counts are its run alone's on
/// a fresh server, the engine's passes alternate slots while both requests
/// run (one pass a slot a round, in the engine thread's one order — the
/// streams' arrival order is the writers' scheduling, not the engine's), and
/// the two streams overlap in arrival.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_drafted_passes_interleave_across_slots() {
    const N: usize = 40;
    let log = Arc::new(EntryLog::default());
    let calls: Arc<Mutex<Vec<(usize, char)>>> = Arc::new(Mutex::new(Vec::new()));
    let addr = start_n(
        Box::new(DraftGated {
            inner: Box::new(serve::DraftMock::new(4096).with_slots(2)),
            log: Arc::clone(&log),
            calls: Arc::clone(&calls),
        }),
        2,
        None,
        None,
    );
    // Greedy and unbanned, so both requests take drafted passes; `ignore_eos`
    // would ban the stop ids and make every token a plain step.
    let body = |p: &str| {
        json!({"prompt": p, "n_predict": N, "temperature": 0, "cache_prompt": false,
               "return_tokens": true, "stream": true})
    };
    let seen: Arc<Mutex<Vec<SeenChunk>>> = Arc::new(Mutex::new(Vec::new()));
    let mut readers = Vec::new();
    for (i, p) in PROMPTS[..2].iter().enumerate() {
        let (addr, seen, body) = (addr, Arc::clone(&seen), body(p));
        readers.push(std::thread::spawn(move || read_stream(addr, &body, &seen)));
        if i == 0 {
            assert!(
                log.len_reached(1, BOUND),
                "the first prompt never reached the engine"
            );
        } else {
            wait_deferred(addr, 1, BOUND);
        }
    }
    log.release();
    for r in readers {
        r.join().expect("reader");
    }
    let seen = seen.lock().expect("seen").clone();
    let finals: Vec<&Value> = seen
        .iter()
        .filter(|(_, _, last, _)| *last)
        .map(|(_, _, _, v)| v)
        .collect();
    assert_eq!(finals.len(), 2, "the two streams' last chunks");
    let mut slots: Vec<u64> = finals
        .iter()
        .map(|v| v["id_slot"].as_u64().expect("id_slot"))
        .collect();
    slots.sort_unstable();
    assert_eq!(slots, [0, 1], "each request its own slot");
    for p in &PROMPTS[..2] {
        let v = finals
            .iter()
            .find(|v| v["prompt"] == *p)
            .unwrap_or_else(|| panic!("no stream of {p}"));
        let alone = start_n(
            Box::new(serve::DraftMock::new(4096).with_slots(2)),
            2,
            None,
            None,
        );
        let mut want_body = body(p);
        want_body["stream"] = json!(false);
        let want = post(alone, "/completion", &want_body).json();
        assert!(
            want["timings"]["draft_n"].as_u64().is_some_and(|n| n > 0),
            "{p}: the alone run drafted"
        );
        assert_eq!(v["tokens"], want["tokens"], "{p}: {}", v);
        assert_eq!(
            v["timings"]["draft_n"], want["timings"]["draft_n"],
            "{p}: the passes it proposed"
        );
        assert_eq!(
            v["timings"]["draft_n_accepted"], want["timings"]["draft_n_accepted"],
            "{p}: the proposals its target kept"
        );
    }
    // The passes alternate: no two of one slot's passes run in a row while
    // the other slot's request is between its first and last pass.
    let passes = pass_slots(&calls.lock().expect("calls").clone());
    assert!(
        passes.len() >= N,
        "fewer than one pass a token ran: {passes:?}"
    );
    for slot in [0, 1] {
        let other = 1 - slot;
        let other_first = passes.iter().position(|&at| at == other);
        let other_last = passes.iter().rposition(|&at| at == other);
        let mut run = 0;
        for (i, &at) in passes.iter().enumerate() {
            if at != slot {
                run = 0;
                continue;
            }
            run += 1;
            let began = i + 1 - run;
            let live = other_first.is_some_and(|f| f <= i) && other_last.is_some_and(|l| l > began);
            assert!(
                !live || run == 1,
                "slot {slot} ran {run} passes in a row while slot {other} was between its \
                 passes: {passes:?}"
            );
        }
    }
    // The streams overlap in arrival: each stream's last chunk lands after
    // the other's first.
    for slot in slots {
        let other = 1 - slot;
        let (Some(mine_last), Some(theirs_first)) = (
            seen.iter().rposition(|(_, s, _, _)| *s == slot),
            seen.iter().position(|(_, s, _, _)| *s == other),
        ) else {
            panic!("stream {slot} or {other} never sent a chunk");
        };
        assert!(
            mine_last > theirs_first,
            "stream {slot} ended before stream {other} began: the streams did not overlap"
        );
    }
    let per = metric(&get(addr, "/metrics").body, "n_busy_slots_per_decode");
    assert!(
        per > 1.0,
        "the requests never ran one round together: {per}"
    );
}

/// A round of two drafted requests is one `advance_slots` call carrying both
/// rows, not a call a row: while both run, the drafted passes the engine takes
/// come in calls of two rows, and never in calls of more rows than slots. The
/// mock keeps one id on every third pass of a slot and two on the others, so
/// 40 ids take about 24 passes a request, and every round but the head's and
/// the tail's carries both: the bound of ten is under half of them.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_drafted_round_is_one_advance_slots_call_of_every_running_slot() {
    const N: usize = 40;
    let log = Arc::new(EntryLog::default());
    let calls: Arc<Mutex<Vec<(usize, char)>>> = Arc::new(Mutex::new(Vec::new()));
    let addr = start_n(
        Box::new(DraftGated {
            inner: Box::new(serve::DraftMock::new(4096).with_slots(2)),
            log: Arc::clone(&log),
            calls: Arc::clone(&calls),
        }),
        2,
        None,
        None,
    );
    // Greedy and unbanned, as the interleave gate's: both requests pass.
    let body = |p: &str| {
        json!({"prompt": p, "n_predict": N, "temperature": 0, "cache_prompt": false,
               "return_tokens": true})
    };
    let a = post_bg(addr, body(PROMPTS[0]));
    assert!(
        log.len_reached(1, BOUND),
        "the first prompt never reached the engine"
    );
    let b = post_bg(addr, body(PROMPTS[1]));
    wait_deferred(addr, 1, BOUND);
    log.release();
    for r in [a.join().expect("first"), b.join().expect("second")] {
        assert_eq!(r.status, 200, "{}", r.body);
        assert!(
            r.json()["timings"]["draft_n"]
                .as_u64()
                .is_some_and(|n| n > 0),
            "a request that never drafted: {}",
            r.body
        );
    }
    let widths: Vec<usize> = calls
        .lock()
        .expect("calls")
        .iter()
        .filter(|(_, what)| *what == 'w')
        .map(|(rows, _)| *rows)
        .collect();
    let both = widths.iter().filter(|&&w| w == 2).count();
    assert!(
        both >= 10,
        "fewer than ten rounds put both slots' passes in one call: {widths:?}"
    );
    assert!(
        widths.iter().all(|&w| w <= 2),
        "a call of more rows than the engine has slots: {widths:?}"
    );
}

/// A request that samples takes plain steps beside a greedy request's drafted
/// passes: both finish, each request's ids are its run alone's on a fresh
/// server, and the rounds carry both slots.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_mixed_step_and_pass_rows() {
    let log = Arc::new(EntryLog::default());
    let calls: Arc<Mutex<Vec<(usize, char)>>> = Arc::new(Mutex::new(Vec::new()));
    let addr = start_n(
        Box::new(DraftGated {
            inner: Box::new(serve::DraftMock::new(4096).with_slots(2)),
            log: Arc::clone(&log),
            calls: Arc::clone(&calls),
        }),
        2,
        None,
        None,
    );
    // The sampler reads the logits row every token, so that request steps;
    // the greedy one passes.
    let sampled = json!({"prompt": PROMPTS[0], "n_predict": 40, "temperature": 0.8, "seed": 7,
                         "cache_prompt": false, "return_tokens": true});
    let drafted = json!({"prompt": PROMPTS[2], "n_predict": 40, "temperature": 0,
                         "cache_prompt": false, "return_tokens": true});
    let a = post_bg(addr, sampled.clone());
    assert!(
        log.len_reached(1, BOUND),
        "the first prompt never reached the engine"
    );
    let b = post_bg(addr, drafted.clone());
    wait_deferred(addr, 1, BOUND);
    log.release();
    let (ra, rb) = (a.join().expect("sampled"), b.join().expect("drafted"));
    assert_eq!(
        (ra.status, rb.status),
        (200, 200),
        "{} | {}",
        ra.body,
        rb.body
    );
    let (va, vb) = (ra.json(), rb.json());
    assert!(
        vb["timings"]["draft_n"].as_u64().is_some_and(|n| n > 0),
        "the greedy request drafted: {vb}"
    );
    assert!(
        va["timings"].get("draft_n").is_none(),
        "the sampled request takes plain steps, no draft counts: {va}"
    );
    for (v, body) in [(&va, &sampled), (&vb, &drafted)] {
        let alone = start_n(
            Box::new(serve::DraftMock::new(4096).with_slots(2)),
            2,
            None,
            None,
        );
        let want = post(alone, "/completion", body).json();
        assert_eq!(v["tokens"], want["tokens"], "{body}");
    }
    let per = metric(&get(addr, "/metrics").body, "n_busy_slots_per_decode");
    assert!(
        per > 1.0,
        "the requests never ran one round together: {per}"
    );
}

/// A drafting engine of two slots that declares no per-slot draft state is
/// refused past one slot by a message naming `slot_drafts`; one that declares
/// it serves `--parallel 2` (the two gates above).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_drafting_engine_without_slot_drafts_is_refused() {
    let slots = SlotConfig {
        parallel: 2,
        ..SlotConfig::default()
    };
    match Server::bind_with(
        "127.0.0.1:0",
        Box::new(DraftingSlots(serve::DraftMock::new(64))),
        config(None),
        slots,
    ) {
        Err(ServeError::Slots(m)) => assert!(m.contains("slot_drafts"), "{m}"),
        Err(e) => panic!("another error: {e}"),
        Ok(_) => panic!("bound: a drafting engine without slot_drafts serves one slot"),
    }
}

/// What a [`Turned`] engine does when asked for a snapshot.
#[derive(Clone, Copy)]
enum Snap {
    /// The mock's own state.
    Takes,
    /// [`serve::StateError::Unsupported`]: an engine that cannot snapshot.
    Cannot,
    /// A [`serve::StateError::Format`]: a snapshot that fails.
    Fails,
    /// The mock's own state, which a resume then refuses.
    Unresumable,
}

/// An engine of one slot (the mock, unless given another) whose every call
/// goes into `trace` in order (`prefill <text>`, `next`, `advance`, `reset`,
/// `snapshot`, `resume`, and `note <line>` for what the prompt cache did),
/// and whose steps (`next` and `advance`, counted over the engine's life)
/// block where `holds` say: the `k`-th one waits until its latch is
/// released. Its prompt cache is off unless [`Turned::caching`].
struct Turned {
    inner: Box<dyn Engine>,
    trace: Arc<Mutex<Vec<String>>>,
    holds: Vec<(usize, Arc<super::Latch>)>,
    steps: usize,
    snap: Snap,
    cache: u64,
}

impl Turned {
    fn new(snap: Snap, holds: &[(usize, &Arc<super::Latch>)]) -> (Turned, Arc<Mutex<Vec<String>>>) {
        Turned::over(Box::new(MockEngine::new(4096)), snap, holds)
    }

    fn over(
        inner: Box<dyn Engine>,
        snap: Snap,
        holds: &[(usize, &Arc<super::Latch>)],
    ) -> (Turned, Arc<Mutex<Vec<String>>>) {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let t = Turned {
            inner,
            trace: Arc::clone(&trace),
            holds: holds.iter().map(|&(k, l)| (k, Arc::clone(l))).collect(),
            steps: 0,
            snap,
            cache: 0,
        };
        (t, trace)
    }

    /// The same engine with a prompt cache of `bytes`.
    fn caching(self, bytes: u64) -> Turned {
        Turned {
            cache: bytes,
            ..self
        }
    }

    fn log(&self, what: String) {
        self.trace.lock().expect("trace").push(what);
    }

    /// One more step, held where `holds` say.
    fn step(&mut self, what: &str) {
        self.steps += 1;
        self.log(what.to_owned());
        if let Some((_, latch)) = self.holds.iter().find(|(k, _)| *k == self.steps) {
            let mut g = latch.state.lock().expect("latch");
            g.0 = true;
            latch.cv.notify_all();
            while !g.1 {
                g = latch.cv.wait(g).expect("latch");
            }
        }
    }
}

impl Engine for Turned {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.log(format!("prefill {}", self.inner.tokenizer().decode(ids)));
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.step("next");
        self.inner.next(last, out)
    }
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.step("advance");
        self.inner.advance(last, out)
    }
    fn advance_rows(&self) -> usize {
        self.inner.advance_rows()
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.log("reset".to_owned());
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
    fn cache_ram(&self) -> u64 {
        self.cache
    }
    fn save_state(&self, out: &mut dyn std::io::Write) -> Result<SavedState, StateError> {
        self.log("snapshot".to_owned());
        match self.snap {
            Snap::Takes | Snap::Unresumable => self.inner.save_state(out),
            Snap::Cannot => Err(StateError::Unsupported("snapshots")),
            Snap::Fails => Err(StateError::Format(
                "the mock refuses this snapshot".to_owned(),
            )),
        }
    }
    fn restore_state(&mut self, input: &mut dyn std::io::Read) -> Result<SavedState, StateError> {
        self.log("resume".to_owned());
        if let Snap::Unresumable = self.snap {
            return Err(StateError::Format(
                "the mock refuses this resume".to_owned(),
            ));
        }
        self.inner.restore_state(input)
    }
    fn note(&self, note: &CacheNote) {
        self.log(format!("note {note}"));
    }
}

/// The prompt cache's saves a [`Turned`] trace holds, in order: each one's
/// positions and whether its state is a snapshot taken for it (`copied`).
fn saves(trace: &[String]) -> Vec<(usize, bool)> {
    trace
        .iter()
        .filter_map(|e| e.strip_prefix("note cache save "))
        .map(|line| {
            let field = |key: &str| {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(key))
                    .unwrap_or_else(|| panic!("no {key} in the save {line:?}"))
            };
            (
                field("positions=").parse().expect("positions"),
                field("copied=").parse().expect("copied"),
            )
        })
        .collect()
}

/// A server of `n` slots that take `engine` in turns.
fn start_swap(engine: Box<dyn Engine>, n: usize, park: Park) -> SocketAddr {
    start_swap_in(engine, n, park, None)
}

/// [`start_swap`] whose slot actions save and restore in `dir`.
fn start_swap_in(engine: Box<dyn Engine>, n: usize, park: Park, dir: Option<&Path>) -> SocketAddr {
    let swap = SwapEngine::new(engine, n, park).unwrap_or_else(|e| panic!("swap engine: {e}"));
    start_n(Box::new(swap), n, None, dir)
}

/// Room for every state these gates park.
const ROOMY: Park = Park::States { budget: 1 << 20 };

/// The request `body` posted on a thread of its own.
fn post_bg(addr: SocketAddr, body: Value) -> std::thread::JoinHandle<super::common::Reply> {
    std::thread::spawn(move || post(addr, "/completion", &body))
}

/// The lengths of the runs of `next` in `trace`, each run ended by any other
/// call.
fn next_runs(trace: &[String]) -> Vec<usize> {
    let mut runs = vec![0];
    for e in trace {
        if e == "next" {
            *runs.last_mut().expect("a run") += 1;
        } else if runs.last() != Some(&0) {
            runs.push(0);
        }
    }
    runs.retain(|&r| r > 0);
    runs
}

/// A request that takes a slot while another decodes runs its prompt at the
/// next step boundary, the running request's state parked first; both give
/// the ids each gives alone.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_request_preempts_the_running_one_at_a_step_and_both_give_their_alone_ids() {
    const HELD: usize = 5;
    let latch = Arc::new(super::Latch::default());
    let (engine, trace) = Turned::new(Snap::Takes, &[(HELD, &latch)]);
    let addr = start_swap(Box::new(engine), 2, ROOMY);
    let (long, short) = (completion("abcabcab", 40), completion("xyzxyz", 6));
    let a = post_bg(addr, long.clone());
    assert!(
        latch.wait_entered(BOUND),
        "the first request never reached next #{HELD}"
    );
    let b = post_bg(addr, short.clone());
    wait_deferred(addr, 1, BOUND);
    latch.release();
    let (a, b) = (a.join().expect("a"), b.join().expect("b"));
    let alone = super::common::start(4096);
    for (r, body) in [(&a, &long), (&b, &short)] {
        assert_eq!(r.status, 200, "{}", r.body);
        let want = post(alone, "/completion", body).json();
        assert_eq!(r.json()["tokens"], want["tokens"], "{body}: {}", r.body);
    }
    let trace = trace.lock().expect("trace").clone();
    let at = trace
        .iter()
        .position(|e| e == "prefill xyzxy")
        .unwrap_or_else(|| panic!("the second prompt never ran: {trace:?}"));
    let before = &trace[..at];
    let nexts = before.iter().filter(|e| *e == "next").count();
    assert_eq!(
        nexts, HELD,
        "the second prompt waited past the step boundary: {trace:?}"
    );
    let last = before.iter().rposition(|e| e == "next").expect("a next");
    assert!(
        before[last..].iter().any(|e| e == "snapshot"),
        "the running request was not parked before the prompt: {trace:?}"
    );
    assert_eq!(
        trace.iter().filter(|e| *e == "resume").count(),
        1,
        "the first request comes back once: {trace:?}"
    );
}

/// Live requests take the engine in turns of `QUANTUM` tokens, the one that
/// waited longest next, and a prompt that arrives during a turn starts at its
/// end: each run of steps is a whole turn but a request's first, cut by the
/// arrival, and its last.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_live_requests_take_turns_and_a_prompt_waits_for_the_turn_boundary() {
    let (first, during) = (
        Arc::new(super::Latch::default()),
        Arc::new(super::Latch::default()),
    );
    let (engine, trace) = Turned::new(Snap::Takes, &[(3, &first), (80, &during)]);
    let addr = start_swap(Box::new(engine), 3, ROOMY);
    let a = post_bg(addr, completion("abcabcab", 200));
    assert!(
        first.wait_entered(BOUND),
        "the first request never reached next #3"
    );
    let b = post_bg(addr, completion("xyzxyz", 200));
    wait_deferred(addr, 1, BOUND);
    first.release();
    assert!(
        during.wait_entered(BOUND),
        "the turns never reached next #80"
    );
    let c = post_bg(addr, completion("pqpqpq", 70));
    wait_deferred(addr, 1, BOUND);
    during.release();
    for r in [a, b, c] {
        let r = r.join().expect("request");
        assert_eq!(r.status, 200, "{}", r.body);
    }
    let q = QUANTUM;
    let trace = trace.lock().expect("trace").clone();
    // A 3 (preempted by B), B q, A q (next #80 inside it), C q, B q, A q,
    // C 6 (its last), B q, A q, B 8 (its last), A 5 (its last).
    assert_eq!(
        next_runs(&trace),
        [3, q, q, q, q, q, 6, q, q, 8, 5],
        "{trace:?}"
    );
}

/// Requests that take slots together start shortest prompt first, the next
/// at the first's turn boundary or end, not at its first step.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_requests_that_take_slots_together_start_shortest_first() {
    let latch = Arc::new(super::Latch::default());
    // Next #2 is the first request's last: the two that wait take slots
    // together, and it ends before either starts.
    let (engine, trace) = Turned::new(Snap::Takes, &[(2, &latch)]);
    let addr = start_swap(Box::new(engine), 3, ROOMY);
    let a = post_bg(addr, completion("abcabcab", 2));
    assert!(
        latch.wait_entered(BOUND),
        "the first request never reached next #2"
    );
    let long = post_bg(addr, completion("pqrstuvwpqrstuvw", 3));
    wait_deferred(addr, 1, BOUND);
    let short = post_bg(addr, completion("mnm", 3));
    wait_deferred(addr, 2, BOUND);
    latch.release();
    for r in [a, long, short] {
        assert_eq!(r.join().expect("request").status, 200);
    }
    let trace = trace.lock().expect("trace").clone();
    let prompts: Vec<&str> = trace
        .iter()
        .filter_map(|e| e.strip_prefix("prefill "))
        .collect();
    assert_eq!(
        prompts,
        ["abcabca", "mn", "pqrstuvwpqrstuv"],
        "the shorter prompt that arrived second starts first"
    );
    let at = |p: &str| trace.iter().position(|e| e == p).expect("a prefill");
    let between = &trace[at("prefill mn")..at("prefill pqrstuvwpqrstuv")];
    assert_eq!(
        between.iter().filter(|e| *e == "next").count(),
        3,
        "the longer prompt waits for the shorter request's three tokens: {trace:?}"
    );
}

/// A live request whose state cannot be parked — a snapshot past the park
/// budget, or one the engine fails — refuses the newcomer with a named 503
/// and `Retry-After`; the running request finishes with its alone ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_state_that_cannot_be_parked_refuses_the_newcomer_by_name() {
    let alone = super::common::start(4096);
    let long = completion("abcabcab", 20);
    let want = post(alone, "/completion", &long).json();
    // The mock's state of ten positions is 56 bytes.
    let cases = [
        (
            Snap::Takes,
            Park::States { budget: 32 },
            "the park budget of 32 bytes",
        ),
        (Snap::Fails, ROOMY, "the mock refuses this snapshot"),
    ];
    for (snap, park, why) in cases {
        let latch = Arc::new(super::Latch::default());
        let (engine, _) = Turned::new(snap, &[(3, &latch)]);
        let addr = start_swap(Box::new(engine), 2, park);
        let a = post_bg(addr, long.clone());
        assert!(latch.wait_entered(BOUND), "{why}: never reached next #3");
        let b = post_bg(addr, completion("xyzxyz", 6));
        wait_deferred(addr, 1, BOUND);
        latch.release();
        let b = b.join().expect("b");
        assert_error(&b, 503, "unavailable_error", why);
        assert_error(&b, 503, "unavailable_error", "slot 0 cannot be parked");
        assert_eq!(b.header("Retry-After"), Some("1"), "{why}: {:?}", b.headers);
        let a = a.join().expect("a");
        assert_eq!(a.status, 200, "{why}: {}", a.body);
        assert_eq!(a.json()["tokens"], want["tokens"], "{why}");
        let m = get(addr, "/metrics").body;
        assert_eq!(metric(&m, "swap_refusals_total"), 1.0, "{why}");
    }
}

/// An engine that cannot snapshot takes turns by its ids: a parked request's
/// ids are fed again at its next turn, and both requests give their alone
/// ids; `/metrics` counts the positions fed again.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_the_re_prefill_fallback_gives_the_alone_ids() {
    let latch = Arc::new(super::Latch::default());
    let (engine, trace) = Turned::new(Snap::Cannot, &[(3, &latch)]);
    let addr = start_swap(Box::new(engine), 2, Park::Ids);
    let bodies = [completion("abcabcab", 100), completion("xyzxyz", 100)];
    let a = post_bg(addr, bodies[0].clone());
    assert!(
        latch.wait_entered(BOUND),
        "the first request never reached next #3"
    );
    let b = post_bg(addr, bodies[1].clone());
    wait_deferred(addr, 1, BOUND);
    latch.release();
    let alone = super::common::start(4096);
    for (r, body) in [a, b].into_iter().zip(&bodies) {
        let r = r.join().expect("request");
        assert_eq!(r.status, 200, "{}", r.body);
        let want = post(alone, "/completion", body).json();
        assert_eq!(r.json()["tokens"], want["tokens"], "{body}");
    }
    let trace = trace.lock().expect("trace").clone();
    assert!(
        !trace.iter().any(|e| e == "snapshot" || e == "resume"),
        "an engine of ids is never asked for a snapshot: {trace:?}"
    );
    let fed = metric(&get(addr, "/metrics").body, "swap_reprefill_tokens_total");
    assert!(fed > 0.0, "no position was fed again");
}

/// `/slots` names each slot's turn: the request whose state is parked, the
/// one running, and idle once both end; `/metrics` counts the switches and
/// states the parked bytes beside the budget.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slots_and_metrics_state_a_parked_request() {
    let (held, during) = (
        Arc::new(super::Latch::default()),
        Arc::new(super::Latch::default()),
    );
    // Next #3 is the first request's; #5 is the second's first step.
    let (engine, _) = Turned::new(Snap::Takes, &[(3, &held), (5, &during)]);
    let addr = start_swap(Box::new(engine), 2, ROOMY);
    let a = post_bg(addr, completion("abcabcab", 8));
    assert!(
        held.wait_entered(BOUND),
        "the first request never reached next #3"
    );
    let b = post_bg(addr, completion("xyzxyz", 4));
    wait_deferred(addr, 1, BOUND);
    held.release();
    assert!(
        during.wait_entered(BOUND),
        "the second request never stepped"
    );
    let turns = |addr| -> Vec<Value> {
        get(addr, "/slots")
            .json()
            .as_array()
            .expect("a list")
            .iter()
            .map(|s| json!([s["turn"], s["is_processing"]]))
            .collect()
    };
    let mid = turns(addr);
    let m = get(addr, "/metrics").body;
    during.release();
    assert_eq!(
        mid,
        [json!(["parked", true]), json!(["running", true])],
        "the first request is parked while the second runs"
    );
    assert_eq!(
        metric(&m, "swap_parked_bytes"),
        56.0,
        "the first request's ten positions: 16 + 4 · 10 bytes"
    );
    assert_eq!(
        metric(&m, "swap_park_budget_bytes"),
        1_048_576.0,
        "the budget the server was given: ROOMY's"
    );
    for r in [a, b] {
        assert_eq!(r.join().expect("request").status, 200);
    }
    assert_eq!(
        turns(addr),
        [json!(["idle", false]), json!(["idle", false])]
    );
    let m = get(addr, "/metrics").body;
    assert_eq!(metric(&m, "swaps_total"), 2.0, "parked once, back once");
}

/// A drafting engine whose slots take turns serves `--parallel 2`, and its
/// greedy requests keep drafting through a preemption: their ids are their
/// alone ids and their timings count drafted ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slots_that_take_turns_keep_the_draft() {
    let latch = Arc::new(super::Latch::default());
    let draft = Box::new(serve::DraftMock::new(4096));
    let (engine, trace) = Turned::over(draft, Snap::Cannot, &[(3, &latch)]);
    let addr = start_swap(Box::new(engine), 2, Park::Ids);
    let bodies = [completion("abcabcab", 30), completion("xyzxyz", 30)];
    let a = post_bg(addr, bodies[0].clone());
    assert!(
        latch.wait_entered(BOUND),
        "the first request never reached step #3"
    );
    let b = post_bg(addr, bodies[1].clone());
    wait_deferred(addr, 1, BOUND);
    latch.release();
    let alone = super::common::start(4096);
    for (r, body) in [a, b].into_iter().zip(&bodies) {
        let r = r.join().expect("request");
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let want = post(alone, "/completion", body).json();
        assert_eq!(v["tokens"], want["tokens"], "{body}");
        assert!(
            v["timings"]["draft_n"].as_u64().is_some_and(|n| n > 0),
            "{body}: no drafted id: {}",
            v["timings"]
        );
    }
    let trace = trace.lock().expect("trace").clone();
    let first_b = trace
        .iter()
        .position(|e| e == "prefill xyzxy")
        .unwrap_or_else(|| panic!("the second prompt never ran: {trace:?}"));
    assert!(
        trace[..first_b].iter().filter(|e| *e == "advance").count() == 2,
        "the second prompt did not preempt the first's passes at a step: {trace:?}"
    );
}

/// A parked state the engine does not take back ends its request with a
/// named error and nothing else: the other request finishes with its alone
/// ids, and the server serves on.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_state_that_does_not_resume_ends_its_request_by_name() {
    let latch = Arc::new(super::Latch::default());
    let (engine, _) = Turned::new(Snap::Unresumable, &[(3, &latch)]);
    let addr = start_swap(Box::new(engine), 2, ROOMY);
    let short = completion("xyzxyz", 6);
    let a = post_bg(addr, completion("abcabcab", 40));
    assert!(
        latch.wait_entered(BOUND),
        "the first request never reached next #3"
    );
    let b = post_bg(addr, short.clone());
    wait_deferred(addr, 1, BOUND);
    latch.release();
    let b = b.join().expect("b");
    assert_eq!(b.status, 200, "{}", b.body);
    let want = post(super::common::start(4096), "/completion", &short).json();
    assert_eq!(b.json()["tokens"], want["tokens"]);
    let a = a.join().expect("a");
    assert_error(
        &a,
        500,
        "server_error",
        "the parked state of slot 0 did not resume: the mock refuses this resume",
    );
    assert_eq!(get(addr, "/health").json()["status"], "ok");
    let after = post(addr, "/completion", &short);
    assert_eq!(after.status, 200, "{}", after.body);
}

/// The pair a turns test opens with: `first` takes slot 0 and is held at its
/// third step until `second`, posted then, has queued for slot 1; both answer
/// 200 on the slots they took.
fn concurrent_pair(addr: SocketAddr, latch: &super::Latch, first: Value, second: Value) {
    let a = post_bg(addr, first);
    assert!(
        latch.wait_entered(BOUND),
        "the first request never reached next #3"
    );
    let b = post_bg(addr, second);
    wait_deferred(addr, 1, BOUND);
    latch.release();
    let answers = [a.join().expect("first"), b.join().expect("second")].map(|r| {
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()
    });
    assert_eq!(
        [&answers[0]["id_slot"], &answers[1]["id_slot"]],
        [&json!(0), &json!(1)],
        "the pair takes both slots"
    );
}

/// Of the free slots that share as much of its prompt, a request takes the
/// one whose state the engine holds: after a concurrent pair, two requests in
/// a row that share only a header with every slot move no state.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_lone_request_takes_the_engines_slot_among_equals() {
    let latch = Arc::new(super::Latch::default());
    let (engine, _) = Turned::new(Snap::Takes, &[(3, &latch)]);
    let addr = start_swap(Box::new(engine), 2, ROOMY);
    concurrent_pair(
        addr,
        &latch,
        completion("sys:abcabcab", 40),
        completion("sys:xyzxyz", 6),
    );
    let swaps = || metric(&get(addr, "/metrics").body, "swaps_total");
    assert_eq!(
        swaps(),
        2.0,
        "the pair: the first parked for the second, then put back"
    );
    // The second ended first, so the engine holds the first's slot, 0.
    let lone: Vec<Value> = ["sys:pqrpqr", "sys:mnomno"]
        .iter()
        .map(|p| post(addr, "/completion", &completion(p, 6)).json())
        .collect();
    assert_eq!(
        swaps(),
        2.0,
        "a lone request whose prompt every slot shares as much of moves no state"
    );
    for v in &lone {
        assert_eq!(v["id_slot"], 0, "the engine's slot: {v}");
        assert_eq!(v["timings"]["cache_n"], 4, "the header kept: {v}");
    }
}

/// Each slot's `n_past` in `/slots`.
fn n_pasts(addr: SocketAddr) -> Vec<Value> {
    get(addr, "/slots")
        .json()
        .as_array()
        .expect("a list")
        .iter()
        .map(|s| s["n_past"].clone())
        .collect()
}

/// An idle slot whose state the engine drops holds nothing after: under the
/// re-prefill fallback the engine leaves the slot of a finished request
/// empty, `/slots` shows it empty, and a request that shares more with what
/// it held than with the engine's slot takes the engine's slot and keeps the
/// header it shares there.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_slot_the_engine_emptied_holds_nothing_for_the_next_request() {
    let latch = Arc::new(super::Latch::default());
    let (engine, _) = Turned::new(Snap::Cannot, &[(3, &latch)]);
    let addr = start_swap(Box::new(engine), 2, Park::Ids);
    concurrent_pair(
        addr,
        &latch,
        completion("system:abcabcab", 40),
        completion("system:pqrpqr", 6),
    );
    // The second ended first; the engine went back to the first's slot and
    // dropped the second's ids.
    let n_past = n_pasts(addr);
    assert_eq!(n_past[1], 0, "the slot the engine emptied: {n_past:?}");
    // Twelve ids shared with what slot 1 held, seven with slot 0's.
    let v = post(addr, "/completion", &completion("system:pqrpqzz", 4)).json();
    assert_eq!(
        v["id_slot"], 0,
        "the slot that holds the shared header: {v}"
    );
    assert_eq!(v["timings"]["cache_n"], 7, "the header kept: {v}");
}

/// An erase names its slot alone: under the re-prefill fallback, erasing a
/// slot the engine does not hold leaves the engine's slot its ids, and a
/// later turn of that slot's request keeps all of them with nothing fed
/// again.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_an_erase_of_another_slot_leaves_the_engines_slot_its_ids() {
    let dir = super::common::fresh_dir("slots-erase-ids");
    let (engine, _) = Turned::new(Snap::Cannot, &[]);
    let addr = start_swap_in(Box::new(engine), 2, Park::Ids, Some(&dir));
    let first = post(addr, "/completion", &completion("sys:abcabcab", 4)).json();
    assert_eq!(first["id_slot"], 0, "{first}");
    // Its twelve prompt ids and three of its four tokens fed.
    assert_eq!(n_pasts(addr), [json!(15), json!(0)], "the fixture");
    let erased = call(addr, "POST", "/slots/1?action=erase", None);
    assert_eq!(erased.status, 200, "{}", erased.body);
    assert_eq!(erased.json(), json!({"id_slot": 1, "n_erased": 0}));
    assert_eq!(
        n_pasts(addr),
        [json!(15), json!(0)],
        "the erase of slot 1 left slot 0's ids"
    );
    let mut later = vec![json!("sys:abcabcab")];
    later.extend(first["tokens"].as_array().expect("tokens").iter().cloned());
    later.push(json!("zz"));
    let body = json!({"prompt": later, "n_predict": 4, "temperature": 0});
    let v = post(addr, "/completion", &body).json();
    assert_eq!(
        (&v["id_slot"], &v["timings"]["cache_n"]),
        (&json!(0), &json!(15)),
        "the later turn keeps slot 0's fifteen ids: {v}"
    );
    let fed = metric(&get(addr, "/metrics").body, "swap_reprefill_tokens_total");
    assert_eq!(fed, 0.0, "slot 0's ids stayed on the engine");
    super::common::drop_dir(&dir);
}

/// An erase of a slot whose idle state is parked drops that state where it
/// lies: the engine keeps its own slot's state and no state moves, nor for
/// the next request on the engine's slot.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_an_erase_of_a_parked_slot_moves_no_state() {
    let dir = super::common::fresh_dir("slots-erase-parked");
    let latch = Arc::new(super::Latch::default());
    let (engine, _) = Turned::new(Snap::Takes, &[(3, &latch)]);
    let addr = start_swap_in(Box::new(engine), 2, ROOMY, Some(&dir));
    concurrent_pair(
        addr,
        &latch,
        completion("sys:abcabcab", 40),
        completion("sys:xyzxyz", 6),
    );
    let swaps = || metric(&get(addr, "/metrics").body, "swaps_total");
    let parked = || metric(&get(addr, "/metrics").body, "swap_parked_bytes");
    // The second ended first: the engine went back to slot 0 and parked slot
    // 1's fifteen positions, 16 + 4 · 15 bytes.
    assert_eq!((swaps(), parked()), (2.0, 76.0), "the fixture");
    let erased = call(addr, "POST", "/slots/1?action=erase", None);
    assert_eq!(erased.status, 200, "{}", erased.body);
    assert_eq!(erased.json(), json!({"id_slot": 1, "n_erased": 15}));
    assert_eq!(
        (swaps(), parked()),
        (2.0, 0.0),
        "the erase of slot 1 dropped its parked state and moved none"
    );
    let v = post(addr, "/completion", &completion("sys:abcabcab", 4)).json();
    assert_eq!(
        (&v["id_slot"], &v["timings"]["cache_n"]),
        (&json!(0), &json!(11)),
        "slot 0 keeps the prompt's eleven ids: {v}"
    );
    assert_eq!(
        swaps(),
        2.0,
        "the request on the engine's slot moved no state"
    );
    super::common::drop_dir(&dir);
}

/// What `trace` holds between the last step before `prompt`'s prefill and
/// that prefill: the switch to the prompt's slot and the prompt cache's work
/// on it.
fn before_prefill(trace: &[String], prompt: &str) -> Vec<String> {
    let at = trace
        .iter()
        .position(|e| *e == format!("prefill {prompt}"))
        .unwrap_or_else(|| panic!("no prefill of {prompt:?}: {trace:?}"));
    let from = trace[..at]
        .iter()
        .rposition(|e| e == "next")
        .unwrap_or_else(|| panic!("no step before the prefill of {prompt:?}: {trace:?}"));
    trace[from + 1..at].to_vec()
}

/// After the park of the state the engine leaves (its first call), the
/// states put back and the snapshots taken.
fn past_park(between: &[String]) -> (usize, usize) {
    assert_eq!(
        between.first().map(String::as_str),
        Some("snapshot"),
        "the state the engine leaves is parked first: {between:?}"
    );
    let count = |what: &str| between[1..].iter().filter(|e| *e == what).count();
    (count("resume"), count("snapshot"))
}

/// A request that starts on a slot whose idle state is parked takes that
/// state from the park table as it is. One that keeps none of it costs no
/// copy past the running request's park — no put back, no snapshot — and the
/// prompt cache holds the state under its ids: a later request that returns
/// to it keeps all of it. One that keeps some of it, less than half, puts it
/// back for the cut and takes no snapshot. The cache's saves say which: a
/// handed-back state is no copy, the engine's state a snapshot. Every request
/// gives its alone ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_parked_idle_state_goes_to_the_cache_without_a_copy() {
    let (held, again) = (
        Arc::new(super::Latch::default()),
        Arc::new(super::Latch::default()),
    );
    // Next #3 is A's third step; B's six steps are #4 to #9; #12 is A's
    // third after it is put back.
    let (engine, trace) = Turned::new(Snap::Takes, &[(3, &held), (12, &again)]);
    let addr = start_swap(Box::new(engine.caching(1 << 20)), 2, ROOMY);
    let bodies = [
        completion("abcabcab", 40),
        completion("xyzxyz", 6),
        completion("pqrpqr", 6),
    ];
    let a = post_bg(addr, bodies[0].clone());
    assert!(held.wait_entered(BOUND), "A never reached next #3");
    let b = post_bg(addr, bodies[1].clone());
    wait_deferred(addr, 1, BOUND);
    held.release();
    assert!(again.wait_entered(BOUND), "A never reached next #12");
    let b = b.join().expect("b");
    let turns: Vec<Value> = get(addr, "/slots")
        .json()
        .as_array()
        .expect("a list")
        .iter()
        .map(|s| s["turn"].clone())
        .collect();
    assert_eq!(
        turns,
        [json!("running"), json!("idle")],
        "the fixture: A runs on slot 0 and B, ended, left slot 1's state parked"
    );
    let r = post_bg(addr, bodies[2].clone());
    wait_deferred(addr, 1, BOUND);
    again.release();
    let (a, r) = (a.join().expect("a"), r.join().expect("r"));
    let alone = super::common::start(4096);
    for (got, body) in [&a, &b, &r].into_iter().zip(&bodies) {
        assert_eq!(got.status, 200, "{}", got.body);
        let want = post(alone, "/completion", body).json();
        assert_eq!(got.json()["tokens"], want["tokens"], "{body}");
    }
    assert_eq!(r.json()["id_slot"], 1, "R takes B's slot: {}", r.body);
    let between = before_prefill(&trace.lock().expect("trace"), "pqrpq");
    assert_eq!(
        past_park(&between),
        (0, 0),
        "R keeps none of B's state: (put back, snapshots) {between:?}"
    );
    // B's eleven held positions (its prompt and five fed ids), from the
    // prompt cache: the engine holds A's.
    let back = completion("xyzxyzxyzxyzx", 6);
    let v = post(addr, "/completion", &back).json();
    assert_eq!(v["id_slot"], 0, "the engine's slot among equals: {v}");
    assert_eq!(v["timings"]["cache_n"], 11, "B's state kept whole: {v}");
    let want = post(alone, "/completion", &back).json();
    assert_eq!(v["tokens"], want["tokens"], "{v}");
    // Two ids of R's state, less than half of its eleven: its slot takes the
    // request over the engine's.
    let short = completion("pqmnmnmn", 6);
    let v = post(addr, "/completion", &short).json();
    assert_eq!(
        (&v["id_slot"], &v["timings"]["cache_n"]),
        (&json!(1), &json!(2)),
        "R's slot, its two shared ids kept: {v}"
    );
    let want = post(alone, "/completion", &short).json();
    assert_eq!(v["tokens"], want["tokens"], "{v}");
    let trace = trace.lock().expect("trace").clone();
    let between = before_prefill(&trace, "mnmnm");
    assert_eq!(
        past_park(&between),
        (1, 0),
        "a request that keeps two ids of R's state: (put back, snapshots) {between:?}"
    );
    // B's eleven positions handed back for R; A's forty-seven (its prompt
    // and thirty-nine fed ids) the engine's, for B back; R's eleven handed
    // back for the short request.
    assert_eq!(
        saves(&trace),
        [(11, false), (47, true), (11, false)],
        "the saves' positions and copies: {trace:?}"
    );
}

/// The idle states slots that take turns stop parking go into the prompt
/// cache, so their conversations come back whole: on three slots, B's idle
/// state leaves the table to make room for A's park when C starts (handed
/// over as it is), and C's finds no room beside A's when it ends (the
/// snapshot its park took). B back and C back each keep the eleven positions
/// their slots held, with their alone ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_states_the_turns_let_go_reach_the_prompt_cache() {
    // A mock state of n positions is 16 + 4n bytes. A parks at ten positions
    // when B starts (56) and B's eleven park beside it when B ends (60, 116
    // in all); A parks at thirteen when C starts (68), which B's 60 beside it
    // pass (128), and C's eleven do not fit beside A's when C ends.
    const BUDGET: u64 = 120;
    let bytes = |n: u64| 16 + 4 * n;
    assert!(
        bytes(10) + bytes(11) <= BUDGET && bytes(13) + bytes(11) > BUDGET,
        "the fixture's budget"
    );
    let (held, again) = (
        Arc::new(super::Latch::default()),
        Arc::new(super::Latch::default()),
    );
    // Next #3 is A's third step; B's six steps are #4 to #9; #12 is A's
    // third after it is put back.
    let (engine, trace) = Turned::new(Snap::Takes, &[(3, &held), (12, &again)]);
    let addr = start_swap(
        Box::new(engine.caching(1 << 20)),
        3,
        Park::States { budget: BUDGET },
    );
    let bodies = [
        completion("abcabcab", 40),
        completion("xyzxyz", 6),
        completion("pqrpqr", 6),
    ];
    let a = post_bg(addr, bodies[0].clone());
    assert!(held.wait_entered(BOUND), "A never reached next #3");
    let b = post_bg(addr, bodies[1].clone());
    wait_deferred(addr, 1, BOUND);
    held.release();
    assert!(again.wait_entered(BOUND), "A never reached next #12");
    let c = post_bg(addr, bodies[2].clone());
    wait_deferred(addr, 1, BOUND);
    again.release();
    let alone = super::common::start(4096);
    for (slot, (r, body)) in [a, b, c].into_iter().zip(&bodies).enumerate() {
        let r = r.join().expect("request");
        assert_eq!(r.status, 200, "{}", r.body);
        let r = r.json();
        assert_eq!(
            r["id_slot"], slot,
            "the fixture: A, B and C on slots 0 to 2: {r}"
        );
        let want = post(alone, "/completion", body).json();
        assert_eq!(r["tokens"], want["tokens"], "{body}");
    }
    let let_go = saves(&trace.lock().expect("trace"));
    for back in ["xyzxyzxyzxyzx", "pqrpqrpqrpqrp"] {
        let body = completion(back, 6);
        let v = post(addr, "/completion", &body).json();
        assert_eq!(
            v["timings"]["cache_n"], 11,
            "{back}: the slot's state kept whole: {v}"
        );
        let want = post(alone, "/completion", &body).json();
        assert_eq!(v["tokens"], want["tokens"], "{back}");
    }
    assert_eq!(
        let_go,
        [(11, false), (11, true)],
        "B's state handed over as it left, then C's snapshot that found no room"
    );
}
