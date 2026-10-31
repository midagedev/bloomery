//! Gate: a prompt that yields. Over a two-slot mock that names a prompt
//! quantum ([`serve::Engine::prompt_quantum`]) and logs every engine call in
//! order: a long prompt that takes a slot while the other decodes runs as calls
//! of a quantum, a decode round of the other slot between each two, and both
//! requests give the ids each gives alone; a prompt alone, and every prompt
//! on an engine that names no quantum, runs in one go as before; the cuts fall
//! on the positions' quantum grid and at each message mark, wherever the call
//! starts and whatever the load; the slot holds after an interleaved prompt what an uninterrupted
//! one leaves; a client gone between the calls ends its request, the slot
//! freed and the other stream on; `/slots` shows the prompt's positions as its
//! calls run; and `prompt_ms` is the prompt's own engine time. A constrained
//! request's run of fixed ids is a call of its own the same way, and `/slots`
//! shows its positions as its calls run.

use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use serve::{
    CacheNote, Decoder, Engine, EngineError, MockEngine, MockTokenizer, SavedState, StateError,
    Tokenizer,
};

use super::{BOUND, completion, post_bg, start_n};
use crate::common::{Reply, get, post};
use crate::{Latch, wait_deferred};

/// The quantum the gates name.
const Q: usize = 4;
/// A request that decodes for a while: the mock's echo of it never ends.
const A: &str = "abcabcab";
/// 21 ids, so a prompt call of 20, five quanta; its last id has come before,
/// so the mock's echo of it goes on.
const B5: &str = "bcdefghijklmnopqrstub";
/// 21 ids with a message mark (`<｜User｜>`, one id) at position 9, off the
/// quantum's grid.
const MARKED: &str = "bcdefghij<｜User｜>klmnopqrstb";

/// The mock's vocabulary with `<｜User｜>` (id 2) opening a message
/// ([`Tokenizer::user_start`]), so a prompt carrying it has a mark its prompt
/// call may be cut at.
struct Marked;

impl Tokenizer for Marked {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }
    fn guard<'a>(&self, text: &'a str, mark: &str) -> std::borrow::Cow<'a, str> {
        MockTokenizer.guard(text, mark)
    }
    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        MockTokenizer.encode_marked(text, mark)
    }
    fn decode(&self, ids: &[u32]) -> String {
        MockTokenizer.decode(ids)
    }
    fn decoder(&self) -> Box<dyn Decoder> {
        MockTokenizer.decoder()
    }
    fn bos(&self) -> Option<u32> {
        MockTokenizer.bos()
    }
    fn eos(&self) -> u32 {
        MockTokenizer.eos()
    }
    fn add_bos(&self) -> bool {
        MockTokenizer.add_bos()
    }
    fn n_vocab(&self) -> usize {
        MockTokenizer.n_vocab()
    }
    fn user_start(&self) -> Vec<u32> {
        vec![2]
    }
}

/// An engine call, in the order the engine thread made it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    /// A `prefill` of `n` ids on `slot`.
    Pre { slot: usize, n: usize },
    /// A `next` on `slot`: a step of one slot, or a row of a step of several
    /// (the trait's default `step_slots`, a select and a `next` a row).
    Next { slot: usize },
    /// The note of a prompt call the engine cut at its marks.
    Split {
        first: usize,
        end: usize,
        at: Vec<usize>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Pre,
    Next,
}

/// What a call waits for before it runs.
enum Then {
    /// Until the latch is released.
    Hold(Arc<Latch>),
    Sleep(Duration),
}

/// The two-slot mock behind [`Marked`] naming `quantum` (`None`: none), every
/// call logged ([`Call`]), every mark it is offered cut at. `at` holds the
/// `k`-th call (counted from 1 on its slot) of a kind; `slow` sleeps before
/// every `next` on its slot.
struct Yielding {
    inner: MockEngine,
    log: Arc<Mutex<Vec<Call>>>,
    at: Vec<(Kind, usize, usize, Then)>,
    slow: Option<(usize, Duration)>,
    counts: [[usize; 4]; 2],
    cur: usize,
}

impl Yielding {
    fn new(quantum: Option<usize>, log: &Arc<Mutex<Vec<Call>>>) -> Yielding {
        let inner = MockEngine::new(4096).with_slots(2);
        Yielding {
            inner: match quantum {
                Some(q) => inner.with_prompt_quantum(q),
                None => inner,
            },
            log: Arc::clone(log),
            at: Vec::new(),
            slow: None,
            counts: [[0; 4]; 2],
            cur: 0,
        }
    }

    /// The call `kind` arrives on the selected slot: what waits at its count.
    fn arrive(&mut self, kind: Kind) {
        let n = &mut self.counts[kind as usize][self.cur];
        *n += 1;
        let n = *n;
        for (k, slot, at, then) in &self.at {
            if *k != kind || *slot != self.cur || *at != n {
                continue;
            }
            match then {
                Then::Hold(latch) => {
                    let mut g = latch.state.lock().expect("latch");
                    g.0 = true;
                    latch.cv.notify_all();
                    while !g.1 {
                        g = latch.cv.wait(g).expect("latch");
                    }
                }
                Then::Sleep(d) => std::thread::sleep(*d),
            }
        }
    }
}

impl Engine for Yielding {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::new(Marked)
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.arrive(Kind::Pre);
        self.log.lock().expect("log").push(Call::Pre {
            slot: self.cur,
            n: ids.len(),
        });
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.arrive(Kind::Next);
        if let Some((slot, d)) = self.slow
            && slot == self.cur
        {
            std::thread::sleep(d);
        }
        self.log
            .lock()
            .expect("log")
            .push(Call::Next { slot: self.cur });
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
        self.inner.select_slot(slot)?;
        self.cur = slot;
        Ok(())
    }
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        self.inner.save_state(out)
    }
    fn prefill_splits(&self, _first: usize, _end: usize, marks: &[usize]) -> Vec<usize> {
        marks.to_vec()
    }
    fn prompt_quantum(&self) -> Option<std::num::NonZeroUsize> {
        self.inner.prompt_quantum()
    }
    fn note(&self, note: &CacheNote) {
        if let CacheNote::Split { first, end, at } = note {
            self.log.lock().expect("log").push(Call::Split {
                first: *first,
                end: *end,
                at: at.clone(),
            });
        }
    }
}

/// A two-slot server on `engine` (slot files in `dir`, if any) and its log.
fn serve(engine: Yielding, dir: Option<&std::path::Path>) -> (SocketAddr, Arc<Mutex<Vec<Call>>>) {
    let log = Arc::clone(&engine.log);
    (start_n(Box::new(engine), 2, None, dir), log)
}

/// [`Yielding`] naming [`Q`], whose `k`-th `next` on `slot` holds on the
/// latch it returns: the request decoding on `slot` stands there while a
/// later one queues.
fn holding(slot: usize, k: usize, log: &Arc<Mutex<Vec<Call>>>) -> (Yielding, Arc<Latch>) {
    let latch = Arc::new(Latch::default());
    let mut e = Yielding::new(Some(Q), log);
    e.at.push((Kind::Next, slot, k, Then::Hold(Arc::clone(&latch))));
    (e, latch)
}

fn new_log() -> Arc<Mutex<Vec<Call>>> {
    Arc::new(Mutex::new(Vec::new()))
}

fn calls(log: &Arc<Mutex<Vec<Call>>>) -> Vec<Call> {
    log.lock().expect("log").clone()
}

/// The lengths of `slot`'s prompt calls, in order.
fn pieces(calls: &[Call], slot: usize) -> Vec<usize> {
    calls
        .iter()
        .filter_map(|c| match *c {
            Call::Pre { slot: s, n } if s == slot => Some(n),
            _ => None,
        })
        .collect()
}

fn splits(calls: &[Call]) -> Vec<Call> {
    calls
        .iter()
        .filter(|c| matches!(c, Call::Split { .. }))
        .cloned()
        .collect()
}

/// A's request on the slot `latch` holds, then `prompt`'s queued behind that
/// hold and the hold released: the second takes the other slot while the
/// first decodes. Returns both replies.
fn beside(addr: SocketAddr, latch: &Latch, prompt: &str, n: usize) -> (Reply, Reply) {
    let a = post_bg(addr, completion(A, 40));
    assert!(latch.wait_entered(BOUND), "A never decoded");
    let b = post_bg(addr, completion(prompt, n));
    wait_deferred(addr, 1, BOUND);
    latch.release();
    let (a, b) = (a.join().expect("a"), b.join().expect("b"));
    assert_eq!((a.status, b.status), (200, 200), "{} | {}", a.body, b.body);
    (a, b)
}

/// The ids `prompt` gives alone, on a one-slot mock.
fn alone(prompt: &str, n: usize) -> Value {
    let addr = crate::common::start(4096);
    post(addr, "/completion", &completion(prompt, n)).json()["tokens"].clone()
}

/// The ids a request holds reach its streaming client before the engine runs
/// another request's prompt call: with A decoding and B and C queued behind
/// it, A's stream carries two ids before B's first prompt call, and B's
/// stream its first id before the call of C's prompt that follows B's last,
/// for a B prompt of one call (with and without a quantum) and of five.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_decode_answer_reaches_the_client_before_a_new_prompt_starts() {
    for (quantum, prompt, call) in [(None, "abab", 1), (Some(Q), "abab", 1), (Some(Q), B5, 5)] {
        let log = new_log();
        let decoded = Arc::new(Latch::default());
        let starting = Arc::new(Latch::default());
        let prompting = Arc::new(Latch::default());
        let mut engine = Yielding::new(quantum, &log);
        engine.inner = engine.inner.with_slots(4);
        engine
            .at
            .push((Kind::Next, 0, 2, Then::Hold(Arc::clone(&decoded))));
        engine
            .at
            .push((Kind::Pre, 1, 1, Then::Hold(Arc::clone(&starting))));
        engine
            .at
            .push((Kind::Pre, 2, call, Then::Hold(Arc::clone(&prompting))));
        let addr = start_n(Box::new(engine), 4, None, None);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let chunks = Arc::clone(&seen);
        let a = std::thread::spawn(move || {
            let mut body = completion(A, 6);
            body["stream"] = json!(true);
            super::read_stream(addr, &body, &chunks);
        });
        assert!(decoded.wait_entered(BOUND), "A never decoded");
        let seen_b = Arc::new(Mutex::new(Vec::new()));
        let chunks = Arc::clone(&seen_b);
        let b = std::thread::spawn(move || {
            let mut body = completion(prompt, 6);
            body["stream"] = json!(true);
            super::read_stream(addr, &body, &chunks);
        });
        wait_deferred(addr, 1, BOUND);
        let c = post_bg(addr, completion(&format!("{B5}{B5}"), 6));
        wait_deferred(addr, 2, BOUND);
        decoded.release();
        let mut deliveries = Vec::new();
        for (latch, want) in [(&starting, (2, 0)), (&prompting, (2, 1))] {
            let entered = latch.wait_entered(BOUND);
            let end = Instant::now() + BOUND;
            let delivered = loop {
                let [a, b] = [&seen, &seen_b].map(|chunks| {
                    chunks
                        .lock()
                        .expect("seen")
                        .iter()
                        .filter(|(_, last, chunk)| {
                            !last && !chunk["content"].as_str().expect("chunk content").is_empty()
                        })
                        .count()
                });
                if (a >= want.0 && b >= want.1) || Instant::now() >= end {
                    break (a, b);
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            latch.release();
            deliveries.push((entered, delivered, want));
        }
        a.join().expect("A stream");
        b.join().expect("B stream");
        let c = c.join().expect("C");
        assert_eq!(c.status, 200, "{}", c.body);
        for (boundary, (entered, delivered, want)) in deliveries.into_iter().enumerate() {
            assert!(entered, "prompt boundary {boundary} never reached");
            assert!(
                delivered.0 >= want.0 && delivered.1 >= want.1,
                "only {delivered:?} tokens before boundary {boundary}, call {call}, quantum {quantum:?}"
            );
        }
    }
}

/// A prompt of five quanta that takes a slot while the other decodes runs as
/// five calls, at least one decode round of the decoding slot between each
/// two, and each request gives the ids it gives alone.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_long_prompt_runs_a_call_a_round_beside_a_decoding_stream() {
    assert_eq!(MockTokenizer.encode(B5).len(), 5 * Q + 1, "the fixture");
    let log = new_log();
    let (engine, latch) = holding(0, 2, &log);
    let (addr, log) = serve(engine, None);
    let (a, b) = beside(addr, &latch, B5, 6);
    let calls = calls(&log);
    assert_eq!(pieces(&calls, 1), [Q; 5], "{calls:?}");
    let at: Vec<usize> = (0..calls.len())
        .filter(|&i| matches!(calls[i], Call::Pre { slot: 1, .. }))
        .collect();
    for w in at.windows(2) {
        assert!(
            calls[w[0]..w[1]].contains(&Call::Next { slot: 0 }),
            "no decode round of A between B's calls at {} and {}: {calls:?}",
            w[0],
            w[1]
        );
    }
    assert_eq!(a.json()["tokens"], alone(A, 40), "A");
    assert_eq!(b.json()["tokens"], alone(B5, 6), "B");
}

/// A prompt alone, no other slot busy, makes the engine calls it makes on an
/// engine that names no quantum: one call, cut at its message mark alone.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_prompt_alone_runs_in_one_go() {
    let runs: Vec<Vec<Call>> = [Some(Q), None]
        .into_iter()
        .map(|q| {
            let (addr, log) = serve(Yielding::new(q, &new_log()), None);
            let r = post(addr, "/completion", &completion(MARKED, 4));
            assert_eq!(r.status, 200, "{}", r.body);
            calls(&log)
        })
        .collect();
    assert_eq!(runs[0], runs[1], "the quantum's engine and the plain one");
    assert_eq!(pieces(&runs[0], 0), [9, 11], "{:?}", runs[0]);
}

/// On an engine that names no quantum, a long prompt that takes a slot while
/// the other decodes still runs in one go.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_an_engine_without_a_quantum_runs_every_prompt_in_one_go() {
    let log = new_log();
    let latch = Arc::new(Latch::default());
    let mut engine = Yielding::new(None, &log);
    engine
        .at
        .push((Kind::Next, 0, 2, Then::Hold(Arc::clone(&latch))));
    let (addr, log) = serve(engine, None);
    let (a, b) = beside(addr, &latch, B5, 6);
    let calls = calls(&log);
    assert_eq!(pieces(&calls, 1), [5 * Q], "{calls:?}");
    assert_eq!(a.json()["tokens"], alone(A, 40), "A");
    assert_eq!(b.json()["tokens"], alone(B5, 6), "B");
}

/// The cuts of an interleaved prompt fall on the multiples of the quantum
/// and at each cut the engine asks for at a message, the same whether it
/// arrives at the decoding slot's first round or its eleventh; the call's
/// note names the mark's cut alone, as the call run in one go does. A
/// prompt whose cache kept a prefix off the grid cuts on the same grid, so
/// its first call runs to the grid's next position.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_the_cuts_fall_on_the_grid_and_the_marks_whatever_the_load() {
    assert_eq!(MockTokenizer.encode(MARKED).len(), 5 * Q + 1, "the fixture");
    for k in [2, 12] {
        let log = new_log();
        let (engine, latch) = holding(0, k, &log);
        let (addr, log) = serve(engine, None);
        beside(addr, &latch, MARKED, 6);
        let calls = calls(&log);
        assert_eq!(
            pieces(&calls, 1),
            [4, 4, 1, 3, 4, 4],
            "the cuts 4, 8, the mark 9, then 12, 16 (B at A's round {k}): {calls:?}"
        );
        assert_eq!(
            splits(&calls),
            [Call::Split {
                first: 0,
                end: 20,
                at: vec![9]
            }],
            "B at A's round {k}"
        );
    }
    // The first request leaves "bcdefghij" on slot 0, A takes the empty slot
    // 1, and the marked prompt slot 0 again: its call starts at 9, past the
    // mark, and the cuts are 12 and 16, the grid's.
    let head = "bcdefghij";
    for k in [2, 12] {
        let log = new_log();
        let (engine, latch) = holding(1, k, &log);
        let (addr, log) = serve(engine, None);
        let r = post(addr, "/completion", &completion(head, 2));
        assert_eq!(r.status, 200, "{}", r.body);
        log.lock().expect("log").clear();
        let (_, b) = beside(addr, &latch, MARKED, 6);
        let v = b.json();
        assert_eq!(v["id_slot"], 0, "{v}");
        assert_eq!(v["timings"]["cache_n"], 9, "{v}");
        let calls = calls(&log);
        assert_eq!(
            pieces(&calls, 0),
            [3, 4, 4],
            "the cuts 12, 16 (B at A's round {k}): {calls:?}"
        );
    }
}

/// After a prompt interleaved with another slot's decode, its slot holds what
/// it holds after the same request alone: the slot's file, its ids and the
/// engine's state, is byte for byte the same, and a later request that
/// extends the prompt keeps as much of it and gives the same ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_an_interleaved_prompt_leaves_the_slot_an_uninterrupted_one_does() {
    let dir = crate::common::fresh_dir("slots-yields");
    let longer = format!("{B5}zz");
    let mut runs: Vec<(Vec<u8>, Value)> = Vec::new();
    for interleaved in [true, false] {
        let log = new_log();
        let (engine, latch) = holding(0, 2, &log);
        let (addr, log) = serve(engine, Some(&dir));
        if interleaved {
            beside(addr, &latch, B5, 3);
        } else {
            latch.release();
            let a = post(addr, "/completion", &completion(A, 3));
            let b = post(addr, "/completion", &completion(B5, 3));
            assert_eq!((a.status, b.status), (200, 200), "{} | {}", a.body, b.body);
        }
        assert_eq!(
            pieces(&calls(&log), 1),
            if interleaved { vec![Q; 5] } else { vec![5 * Q] },
            "the fixture: B's prompt interleaved or not"
        );
        let file = format!("b-{interleaved}.bin");
        let saved = post(addr, "/slots/1?action=save", &json!({ "filename": file }));
        assert_eq!(saved.status, 200, "{}", saved.body);
        let bytes = std::fs::read(dir.join(&file)).expect("the slot file");
        let c = post(addr, "/completion", &completion(&longer, 3)).json();
        assert_eq!(c["id_slot"], 1, "{c}");
        runs.push((bytes, c));
    }
    let ((inter, c_inter), (whole, c_whole)) = (&runs[0], &runs[1]);
    assert!(inter == whole, "slot 1's file after B differs");
    for key in ["cache_n", "prompt_n"] {
        assert_eq!(
            c_inter["timings"][key], c_whole["timings"][key],
            "{key}: {c_inter} | {c_whole}"
        );
    }
    assert_eq!(c_inter["timings"]["cache_n"], 21, "{c_inter}");
    assert_eq!(c_inter["tokens"], c_whole["tokens"]);
    crate::common::drop_dir(&dir);
}

/// A client gone between a prompt's calls ends its request: `/slots` showed
/// the prompt's positions as its calls ran; the calls after the server saw
/// the client go are none (the one in flight and the one its probe ran
/// beside, the lag a token's text has too, then nothing), its last id is
/// never stepped, its slot frees, the decoding stream gives its alone ids and
/// the engine thread serves on.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_client_gone_between_the_calls_ends_its_request() {
    let log = new_log();
    let (mut engine, decoding) = holding(0, 2, &log);
    let third = Arc::new(Latch::default());
    engine
        .at
        .push((Kind::Pre, 1, 3, Then::Hold(Arc::clone(&third))));
    // The fourth call runs beside the probe the server's write fails on; its
    // pause lets that write fail before the probe after it.
    engine
        .at
        .push((Kind::Pre, 1, 4, Then::Sleep(Duration::from_millis(300))));
    let (addr, log) = serve(engine, None);
    let a = post_bg(addr, completion(A, 40));
    assert!(decoding.wait_entered(BOUND), "A never decoded");
    let mut b = TcpStream::connect(addr).expect("connect");
    let body = json!({"prompt": B5, "n_predict": 4, "temperature": 0, "stream": true}).to_string();
    let req = format!(
        "POST /completion HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    b.write_all(req.as_bytes()).expect("write");
    wait_deferred(addr, 1, BOUND);
    decoding.release();
    assert!(
        third.wait_entered(BOUND),
        "B's third call never came: {:?}",
        calls(&log)
    );
    let view = get(addr, "/slots").json()[1].clone();
    assert_eq!(view["is_processing"], true, "{view}");
    assert_eq!(view["n_past"], 2 * Q, "two calls in: {view}");
    // The probes before the second and the third call reach the client,
    // which leaves them unread: its close then resets the connection, and
    // the server's next write fails at once.
    b.set_read_timeout(Some(BOUND)).expect("timeout");
    let deadline = Instant::now() + BOUND;
    let mut buf = [0u8; 4096];
    loop {
        let n = b.peek(&mut buf).expect("peek");
        let got = String::from_utf8_lossy(&buf[..n]);
        if got.matches("data: ").count() >= 2 {
            break;
        }
        assert!(Instant::now() < deadline, "the probes never came: {got}");
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(b);
    std::thread::sleep(Duration::from_millis(50));
    third.release();
    let a = a.join().expect("a");
    assert_eq!(a.status, 200, "{}", a.body);
    assert_eq!(a.json()["tokens"], alone(A, 40), "A");
    let deadline = Instant::now() + BOUND;
    while get(addr, "/slots").json()[1]["is_processing"] != json!(false) {
        assert!(
            Instant::now() < deadline,
            "B's slot never freed: {:?}",
            calls(&log)
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let calls_b = calls(&log);
    assert_eq!(pieces(&calls_b, 1), [Q; 4], "{calls_b:?}");
    assert!(
        !calls_b.contains(&Call::Next { slot: 1 }),
        "B's last id was stepped: {calls_b:?}"
    );
    let e = post(addr, "/completion", &completion(B5, 3));
    assert_eq!(e.status, 200, "{}", e.body);
}

/// `prompt_ms` is the prompt's own engine time: beside a decoding slot whose
/// every step is slow, an interleaved prompt spans four of those steps and
/// its `prompt_ms` stays under one.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_prompt_ms_is_the_prompts_own_engine_time() {
    const SLOW: Duration = Duration::from_millis(50);
    let log = new_log();
    let (mut engine, latch) = holding(0, 2, &log);
    engine.slow = Some((0, SLOW));
    let (addr, log) = serve(engine, None);
    let a = post_bg(addr, completion(A, 12));
    assert!(latch.wait_entered(BOUND), "A never decoded");
    let b = post_bg(addr, completion(B5, 2));
    wait_deferred(addr, 1, BOUND);
    let t0 = Instant::now();
    latch.release();
    let b = b.join().expect("b");
    let wall = t0.elapsed();
    assert_eq!(b.status, 200, "{}", b.body);
    assert_eq!(a.join().expect("a").status, 200);
    let calls = calls(&log);
    assert_eq!(pieces(&calls, 1), [Q; 5], "{calls:?}");
    assert!(wall >= 4 * SLOW, "B spanned no slow rounds: {wall:?}");
    let ms = b.json()["timings"]["prompt_ms"]
        .as_f64()
        .expect("prompt_ms");
    assert!(
        ms < SLOW.as_secs_f64() * 1e3,
        "prompt_ms {ms} counts the rounds between B's calls ({wall:?} in all)"
    );
}

/// A two-slot server on `engine` whose requests that follow a program take
/// the jump arm, or the step arm alone.
fn serve_jump(engine: Yielding, jump: bool) -> (SocketAddr, Arc<Mutex<Vec<Call>>>) {
    let log = Arc::clone(&engine.log);
    let slots = serve::SlotConfig {
        parallel: 2,
        schema_jump: jump,
        ..serve::SlotConfig::default()
    };
    let server =
        serve::Server::bind_with("127.0.0.1:0", Box::new(engine), super::config(None), slots)
            .unwrap_or_else(|e| panic!("bind: {e}"));
    (server.spawn().expect("spawn"), log)
}

/// A request for a document of two booleans. Its answer opens with `{"a": `,
/// the answer's lead the jump arm's prompt call carries, and the run after
/// the first boolean's pick, the rest of the boolean, `, "`, a property name
/// of 24 bytes and `": `, is more than six quanta.
fn document() -> Value {
    let schema = json!({"type": "object", "properties": {
        "a": {"type": "boolean"},
        "abcdefghijklmnopqrstuvwx": {"type": "boolean"}}});
    json!({"prompt": "ab", "n_predict": 100, "return_tokens": true,
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "r", "schema": schema}}})
}

/// A run of fixed ids reaches its streaming client before another request's
/// fill finishes: beside a decoding stream, of two streams of [`document`],
/// the first carries its 24-byte property name while the second's last fill
/// call is held, with and without a quantum, and with a third property after
/// that name or none; both streams give the same valid document.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_schema_fill_reaches_the_client_before_the_next_fill_finishes() {
    for quantum in [None, Some(Q)] {
        for more in [false, true] {
            let log = new_log();
            let decoded = Arc::new(Latch::default());
            let filling = Arc::new(Latch::default());
            let mut engine = Yielding::new(quantum, &log);
            engine.inner = engine.inner.with_slots(4);
            engine
                .at
                .push((Kind::Next, 0, 2, Then::Hold(Arc::clone(&decoded))));
            engine
                .at
                .push((Kind::Next, 2, 2, Then::Hold(Arc::clone(&filling))));
            let addr = start_n(Box::new(engine), 4, None, None);
            let running = post_bg(addr, completion(A, 60));
            assert!(decoded.wait_entered(BOUND), "the decoding slot never ran");
            let mut body = document();
            if more {
                body["response_format"]["json_schema"]["schema"]["properties"]["z"] =
                    json!({"type": "boolean"});
            }
            body["stream"] = json!(true);
            let mut readers = Vec::new();
            let mut streams = Vec::new();
            for queued in 1..=2 {
                let seen = Arc::new(Mutex::new(Vec::new()));
                let chunks = Arc::clone(&seen);
                let body = body.clone();
                readers.push(std::thread::spawn(move || {
                    super::read_stream(addr, &body, &chunks);
                }));
                streams.push(seen);
                wait_deferred(addr, queued, BOUND);
            }
            decoded.release();
            let entered = filling.wait_entered(BOUND);
            let deadline = Instant::now() + BOUND;
            let delivered = loop {
                let text: String = streams[0]
                    .lock()
                    .expect("seen")
                    .iter()
                    .map(|(_, _, chunk)| chunk["content"].as_str().expect("content"))
                    .collect();
                if text.contains("abcdefghijklmnopqrstuvwx") || Instant::now() >= deadline {
                    break text;
                }
                std::thread::sleep(Duration::from_millis(5));
            };
            filling.release();
            for reader in readers {
                reader.join().expect("schema stream");
            }
            let running = running.join().expect("decoding stream");
            assert_eq!(running.status, 200, "{}", running.body);
            assert!(entered, "B's final fill never ran: {:?}", calls(&log));
            assert!(
                delivered.contains("abcdefghijklmnopqrstuvwx"),
                "A's fill waited for B: {delivered:?}, quantum {quantum:?}, more {more}"
            );
            let texts: Vec<String> = streams
                .iter()
                .map(|seen| {
                    let chunks = seen.lock().expect("seen");
                    assert!(chunks.last().expect("last chunk").1, "stream never ended");
                    chunks
                        .iter()
                        .map(|(_, _, chunk)| chunk["content"].as_str().expect("content"))
                        .collect()
                })
                .collect();
            assert_eq!(texts[0], texts[1], "the schema streams differ");
            let value: Value = serde_json::from_str(&texts[0]).expect("schema document");
            for key in ["a", "abcdefghijklmnopqrstuvwx"] {
                assert!(value[key].is_boolean(), "{value}");
            }
            assert_eq!(value["z"].is_boolean(), more, "{value}");
        }
    }
}

/// B's calls on slot 1 in `calls`, cut after each of its steps: the prompt's
/// calls, then a run's (the trait's `next_after` is a prefill and a step).
fn b_rounds(calls: &[Call]) -> Vec<Vec<Call>> {
    let mut out = vec![Vec::new()];
    for c in calls {
        if !matches!(c, Call::Pre { slot: 1, .. } | Call::Next { slot: 1 }) {
            continue;
        }
        out.last_mut().expect("a round").push(c.clone());
        if matches!(c, Call::Next { slot: 1 }) {
            out.push(Vec::new());
        }
    }
    out
}

/// A constrained request that takes a slot while the other decodes runs each
/// run of fixed ids as calls cut on the quantum's grid, a decode round of the
/// other slot between each two, the last of them stepping the run's last id;
/// the answer's lead rides the prompt's calls, cut on the grid as the prompt
/// is. On the jump
/// arm it gives the step arm's ids in fewer calls, and nothing after the last
/// pick: B makes the prompt's calls and one run's.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_run_of_fixed_ids_is_a_call_a_round_beside_a_decoding_stream() {
    let mut ran = Vec::new();
    for jump in [true, false] {
        let log = new_log();
        let (engine, latch) = holding(0, 2, &log);
        let (addr, log) = serve_jump(engine, jump);
        let a = post_bg(addr, completion(A, 60));
        assert!(latch.wait_entered(BOUND), "A never decoded");
        // A prompt of three ids: the run after the first pick starts off
        // the quantum's grid, at position 9.
        let mut body = document();
        body["prompt"] = json!("abc");
        let b = std::thread::spawn(move || post(addr, "/completion", &body));
        wait_deferred(addr, 1, BOUND);
        latch.release();
        let (a, b) = (a.join().expect("a"), b.join().expect("b"));
        assert_eq!((a.status, b.status), (200, 200), "{} | {}", a.body, b.body);
        ran.push((a, b, calls(&log)));
    }
    let ((a_jump, b_jump, jump), (a_step, b_step, step)) = (&ran[0], &ran[1]);
    assert_eq!(a_jump.json()["tokens"], a_step.json()["tokens"], "A");
    assert_eq!(b_jump.json()["tokens"], b_step.json()["tokens"], "B");
    assert_eq!(b_jump.json()["content"], b_step.json()["content"], "B");
    let rounds = b_rounds(jump);
    assert_eq!(
        rounds.len(),
        3,
        "the prompt's calls, one run's, none after: {jump:?}"
    );
    assert!(rounds[2].is_empty(), "a call after the last pick: {jump:?}");
    let pieces_of = |r: &[Call]| -> Vec<usize> {
        r.iter()
            .filter_map(|c| match *c {
                Call::Pre { n, .. } => Some(n),
                _ => None,
            })
            .collect()
    };
    let lead = pieces_of(&rounds[0]);
    let run = pieces_of(&rounds[1]);
    assert!(
        lead.len() > 1,
        "the prompt and its lead are two quanta: {jump:?}"
    );
    assert!(run.len() > 6, "the run is more than six quanta: {jump:?}");
    assert!(lead.iter().chain(&run).all(|&n| n <= Q), "{jump:?}");
    // Every call but a round's last ends on the quantum's grid: B's position
    // after it, counted from its fresh slot, is a multiple of the quantum.
    let mut at = 0;
    let mut off_grid = Vec::new();
    for round in &rounds {
        let mut left = pieces_of(round).len();
        for c in round {
            match *c {
                Call::Pre { n, .. } => {
                    at += n;
                    left -= 1;
                    if left > 0 && !at.is_multiple_of(Q) {
                        off_grid.push(at);
                    }
                }
                Call::Next { .. } => at += 1,
                Call::Split { .. } => {}
            }
        }
    }
    assert!(
        off_grid.is_empty(),
        "calls end off the quantum's grid at {off_grid:?}: {jump:?}"
    );
    // Inside the prompt's calls and inside the run's; the run's first call
    // is the round the prompt's last call joins the running requests in.
    let mut at: Vec<Vec<usize>> = vec![Vec::new()];
    for (i, c) in jump.iter().enumerate() {
        match c {
            Call::Pre { slot: 1, .. } => at.last_mut().expect("a round").push(i),
            Call::Next { slot: 1 } => at.push(Vec::new()),
            _ => {}
        }
    }
    for w in at.iter().flat_map(|r| r.windows(2)) {
        assert!(
            jump[w[0]..w[1]].contains(&Call::Next { slot: 0 }),
            "no decode round of A between B's calls at {} and {}: {jump:?}",
            w[0],
            w[1]
        );
    }
    let on = b_jump.json()["timings"]["constrained"].clone();
    let off = b_step.json()["timings"]["constrained"].clone();
    assert_eq!(on["arm"], "jump");
    assert_eq!(on["lead"], 6, "{on}");
    assert_eq!(
        (
            on["spans"].as_u64(),
            on["next_after_calls"].as_u64(),
            on["next_calls"].as_u64()
        ),
        (Some(1), Some(1), Some(0)),
        "{on}"
    );
    assert_eq!(on["prefill_calls"], run.len() - 1, "{on}");
    assert_eq!(on["span_ids"], run.iter().sum::<usize>() + 1, "{on}");
    assert_eq!(on["decisions"], off["decisions"]);
    assert_eq!(
        (
            off["arm"].as_str(),
            off["spans"].as_u64(),
            off["prefill_calls"].as_u64()
        ),
        (Some("step"), Some(0), Some(0))
    );
    assert!(
        pieces(step, 1).len() == 1,
        "the step arm prefills the prompt alone: {step:?}"
    );
    let b_calls = |c: &[Call]| {
        c.iter()
            .filter(|c| matches!(c, Call::Pre { slot: 1, .. } | Call::Next { slot: 1 }))
            .count()
    };
    assert!(b_calls(jump) < b_calls(step), "{jump:?} | {step:?}");
}

/// `/slots` shows a run's positions as its calls run, as a prompt's: with the
/// second call of B's run in the engine (B's fourth prefill: the prompt and
/// its lead are two calls, then the run's first), the view holds the
/// positions the calls before it left, the ids the engine has been fed on B's
/// slot.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slots_shows_a_runs_positions_as_its_calls_run() {
    let log = new_log();
    let (mut engine, latch) = holding(0, 2, &log);
    let second = Arc::new(Latch::default());
    engine
        .at
        .push((Kind::Pre, 1, 4, Then::Hold(Arc::clone(&second))));
    let (addr, log) = serve_jump(engine, true);
    let a = post_bg(addr, completion(A, 60));
    assert!(latch.wait_entered(BOUND), "A never decoded");
    let b = std::thread::spawn(move || post(addr, "/completion", &document()));
    wait_deferred(addr, 1, BOUND);
    latch.release();
    assert!(
        second.wait_entered(BOUND),
        "B's second call of the run never came: {:?}",
        calls(&log)
    );
    let fed: usize = calls(&log)
        .iter()
        .map(|c| match *c {
            Call::Pre { slot: 1, n } => n,
            Call::Next { slot: 1 } => 1,
            _ => 0,
        })
        .sum();
    assert_eq!(b_rounds(&calls(&log)).len(), 2, "{:?}", calls(&log));
    let view = get(addr, "/slots").json()[1].clone();
    assert_eq!(view["n_past"], fed, "{view}");
    second.release();
    let (a, b) = (a.join().expect("a"), b.join().expect("b"));
    assert_eq!((a.status, b.status), (200, 200), "{} | {}", a.body, b.body);
}
