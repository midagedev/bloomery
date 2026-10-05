//! Gate: a prompt that yields. Over a two-slot mock that names a prompt
//! quantum ([`serve::Engine::prompt_quantum`]) and logs every engine call in
//! order: a long prompt that takes a slot while the other decodes runs as calls
//! of a quantum, a decode round of the other slot between each two, and both
//! requests give the ids each gives alone; a prompt alone, and every prompt
//! on an engine that names no quantum, runs in one go as before; the cuts are
//! counted from the call's start and again from each message mark, whatever
//! the load; the slot holds after an interleaved prompt what an uninterrupted
//! one leaves; a client gone between the calls ends its request, the slot
//! freed and the other stream on; `/slots` shows the prompt's positions as its
//! calls run; and `prompt_ms` is the prompt's own engine time.

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
    fn decode(&self, ids: &[u32]) -> String {
        MockTokenizer.decode(ids)
    }
    fn decoder(&self) -> Box<dyn Decoder> {
        MockTokenizer.decoder()
    }
    fn bos(&self) -> u32 {
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
    counts: [[usize; 2]; 2],
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
            counts: [[0; 2]; 2],
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

/// The cuts of an interleaved prompt are counted from its call's start and
/// again from each cut the engine asks for at a message, the same whether it
/// arrives at the decoding slot's first round or its eleventh; the call's
/// note names the mark's cut alone, as the call run in one go does. A
/// prompt whose cache kept a prefix counts from where its call starts.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_the_cuts_follow_the_call_and_its_marks_whatever_the_load() {
    assert_eq!(MockTokenizer.encode(MARKED).len(), 5 * Q + 1, "the fixture");
    for k in [2, 12] {
        let log = new_log();
        let (engine, latch) = holding(0, k, &log);
        let (addr, log) = serve(engine, None);
        beside(addr, &latch, MARKED, 6);
        let calls = calls(&log);
        assert_eq!(
            pieces(&calls, 1),
            [4, 4, 1, 4, 4, 3],
            "the cuts 4, 8, the mark 9, then 13, 17 (B at A's round {k}): {calls:?}"
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
    // mark, so the cuts are 13 and 17.
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
            [4, 4, 3],
            "the cuts 13, 17 (B at A's round {k}): {calls:?}"
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
