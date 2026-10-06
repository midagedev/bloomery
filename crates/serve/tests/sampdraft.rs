//! Gate: a sampled request on an engine that drafts sampled requests
//! ([`serve::Engine::advance_sampled`]) takes the ids a plain run of the same
//! seed takes, alone in its round, and steps where a pass would read what the
//! loop has not settled: an id it bans, a think close it forces, another busy
//! slot's rows. The drafted mock is [`DraftMock`]; its plain twin is the same
//! mock [`DraftMock::without_sampled`]. Both servers sample with
//! [`penalised`], whose draws read the history, so a pass that hands the
//! sampler another history than a step does takes other ids.

mod common;

use std::net::SocketAddr;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use common::{get, post, start_sampling};
use serde_json::{Value, json};
use serve::mock::MockCall;
use serve::{
    DraftMock, Drafted, Engine, EngineError, Sampler, SamplerFactory, SamplingParams, Tokenizer,
};

/// A prompt whose `a` has four followers, so the sampler branches at every
/// `a` the model echoes.
const BRANCHY: &str = "abacadaeabacada";

const BOUND: Duration = Duration::from_secs(10);

/// The server's reference sampler behind a repetition penalty over the last 8
/// ids of the history it is given (a positive logit divided by 1.3, a negative
/// one multiplied, once an occurrence).
fn penalised() -> SamplerFactory {
    let inner = serve::sampling::reference_factory();
    Arc::new(move |p: &SamplingParams| -> Sampler {
        let mut draw = inner(p);
        let mut row: Vec<f32> = Vec::new();
        Box::new(move |logits: &[f32], history: &[u32]| {
            row.clear();
            row.extend_from_slice(logits);
            for &id in history.iter().rev().take(8) {
                if let Some(l) = row.get_mut(id as usize) {
                    *l = if *l > 0.0 { *l / 1.3 } else { *l * 1.3 };
                }
            }
            draw(&row, history)
        })
    })
}

type Log = Arc<Mutex<Vec<MockCall>>>;

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

/// The calls `log` holds, which it then drops.
fn drain(log: &Log) -> Vec<MockCall> {
    std::mem::take(&mut *log.lock().expect("the call log"))
}

fn count(calls: &[MockCall], kind: MockCall) -> usize {
    calls.iter().filter(|&&c| c == kind).count()
}

/// A server of one slot on `mock`, logged into `log`, sampling with
/// [`penalised`].
fn serve(mock: DraftMock, log: &Log) -> SocketAddr {
    start_sampling(Box::new(mock.logged(Arc::clone(log))), 1, penalised())
}

/// The drafting mock these gates draft with: three ids a pass, two of every
/// three the target's argmax.
fn drafting(ctx: usize) -> DraftMock {
    DraftMock::new(ctx).with_sampled_draft(3, 2, 3)
}

/// A `/completion` of `body` with `return_tokens`, which must be served: its
/// ids and its reply.
fn ids_of(addr: SocketAddr, body: &Value) -> (Vec<u64>, Value) {
    let mut b = body.clone();
    b["return_tokens"] = json!(true);
    let r = post(addr, "/completion", &b);
    assert_eq!(r.status, 200, "{b}: {}", r.body);
    let v = r.json();
    let ids = v["tokens"]
        .as_array()
        .unwrap_or_else(|| panic!("no tokens: {v}"))
        .iter()
        .map(|t| t.as_u64().expect("an id"))
        .collect();
    (ids, v)
}

/// The same request streamed: its text chunks joined, its ids and its last
/// event.
fn streamed_ids(addr: SocketAddr, body: &Value) -> (String, Vec<u64>, Value) {
    let mut b = body.clone();
    b["return_tokens"] = json!(true);
    b["stream"] = json!(true);
    let r = post(addr, "/completion", &b);
    assert_eq!(r.status, 200, "{b}: {}", r.body);
    let events: Vec<Value> = r
        .events()
        .iter()
        .map(|e| serde_json::from_str(e).unwrap_or_else(|err| panic!("{err}: {e}")))
        .collect();
    let text: String = events
        .iter()
        .filter_map(|e| e["content"].as_str())
        .collect();
    let last = events.last().cloned().expect("a final event");
    assert_eq!(
        last["stop"], true,
        "the stream ends on its final event: {last}"
    );
    let ids = last["tokens"]
        .as_array()
        .unwrap_or_else(|| panic!("no tokens: {last}"))
        .iter()
        .map(|t| t.as_u64().expect("an id"))
        .collect();
    (text, ids, last)
}

/// A sampled `/completion` of `prompt`.
fn sampled(prompt: &str, n_predict: usize, temperature: f64, seed: u64) -> Value {
    json!({"prompt": prompt, "n_predict": n_predict, "temperature": temperature, "seed": seed})
}

/// The draft counts of a reply, `(0, 0)` when it has none.
fn draft_counts(v: &Value) -> (u64, u64) {
    let t = &v["timings"];
    (
        t["draft_n"].as_u64().unwrap_or(0),
        t["draft_n_accepted"].as_u64().unwrap_or(0),
    )
}

/// A drafted sampled request takes the plain run's ids at T 0.6, 0.8 and 1.0
/// over many seeds, streamed (its text and ids) and not, its passes cut by
/// `n_predict` and by a stop word inside their kept ids; the passes ran,
/// some of their proposals kept and some taken back, the plain twin ran none,
/// and the sampler branched off the greedy ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_drafted_sampled_ids_are_the_plain_ids() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let (greedy, _) = ids_of(
        plain,
        &json!({"prompt": BRANCHY, "n_predict": 48, "temperature": 0}),
    );
    let (mut proposed, mut accepted, mut branched) = (0, 0, false);
    for temperature in [0.6, 0.8, 1.0] {
        for seed in 1..=8u64 {
            for (n_predict, stop) in [(48, None), (37, None), (48, Some("ae"))] {
                let mut body = sampled(BRANCHY, n_predict, temperature, seed);
                if let Some(s) = stop {
                    body["stop"] = json!([s]);
                }
                let at = format!("T {temperature}, seed {seed}, {body}");
                let (want, w) = ids_of(plain, &body);
                let (got, v) = ids_of(drafted, &body);
                assert_eq!(got, want, "{at}: {v}");
                assert_eq!(v["content"], w["content"], "{at}");
                assert_eq!(v["stop_type"], w["stop_type"], "{at}");
                let (text, ids, last) = streamed_ids(drafted, &body);
                assert_eq!(ids, want, "{at}, streamed: {last}");
                assert_eq!(json!(text), w["content"], "{at}, streamed");
                for d in [&v, &last] {
                    let (n, a) = draft_counts(d);
                    (proposed, accepted) = (proposed + n, accepted + a);
                }
                assert_eq!(
                    draft_counts(&w),
                    (0, 0),
                    "{at}: the plain twin drafted: {w}"
                );
                branched |= n_predict == 48 && stop.is_none() && got[..] != greedy[..];
            }
        }
    }
    assert!(
        0 < accepted && accepted < proposed,
        "the passes kept {accepted} of {proposed} proposed ids: both kept and taken-back rows are \
         needed"
    );
    assert!(branched, "no request sampled off the greedy ids {greedy:?}");
    let (d, p) = (drain(&dlog), drain(&plog));
    assert!(count(&d, MockCall::Sampled) > 0, "no sampled pass ran");
    assert_eq!(count(&p, MockCall::Sampled), 0, "the plain twin ran a pass");
    // The prompt's tail in the penalty window: a presence that pushes every
    // echoed id below the mock's floor moves the draws — the first moved id
    // named below — and the drafted run still takes the plain run's ids, its
    // passes drawing on the same window the plain steps draw on.
    let mut moved = None;
    for seed in 1..=8u64 {
        let neutral = ids_of(plain, &sampled(BRANCHY, 48, 0.8, seed)).0;
        let mut body = sampled(BRANCHY, 48, 0.8, seed);
        body["presence_penalty"] = json!(20.0);
        let at = format!("penalized, seed {seed}, {body}");
        let (want, w) = ids_of(plain, &body);
        let (got, v) = ids_of(drafted, &body);
        assert_eq!(got, want, "{at}: {v}");
        assert_eq!(v["content"], w["content"], "{at}");
        if moved.is_none()
            && let Some(i) = got.iter().zip(&neutral).position(|(g, &n)| g != &n)
        {
            moved = Some((seed, i, got[i], neutral[i]));
        }
    }
    let Some((seed, at, got, was)) = moved else {
        panic!("presence 20 moved no drawn id over the seeds");
    };
    println!("presence 20 moved seed {seed}'s id {at}: {was} -> {got}");
    assert!(
        count(&drain(&dlog), MockCall::Sampled) > 0,
        "the penalized arm's drafted requests passed"
    );
}

/// Near the context's end a sampled pass that would pass it is not run: the
/// positions left take one step each, so the drafted ids end where the plain
/// ones do. Eight contexts in a row put the end at every phase of a pass of
/// four rows.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_sampled_draft_stops_at_the_plain_context_end() {
    for ctx in 40..48 {
        let (dlog, plog) = (log(), log());
        let drafted = serve(drafting(ctx), &dlog);
        let plain = serve(drafting(ctx).without_sampled(), &plog);
        for seed in 1..=3 {
            let body = sampled(BRANCHY, 100, 0.8, seed);
            let (want, w) = ids_of(plain, &body);
            let (got, v) = ids_of(drafted, &body);
            assert_eq!(got, want, "ctx {ctx}, seed {seed}: {v}");
            assert_eq!(w["truncated"], true, "{w}");
            assert_eq!(v["truncated"], true, "{v}");
        }
        assert!(count(&drain(&dlog), MockCall::Sampled) > 0, "ctx {ctx}");
    }
}

/// The sampler's draws run on across a pass, a step and a pass of one
/// request: a think budget lets passes run while it holds a pass's rows,
/// steps (each a draw) below that, forces the close (no draw), and passes
/// resume once the span is closed. Each request's ids are the plain run's,
/// and some request draws in a step between two passes.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_draws_run_on_across_passes_and_steps() {
    // Inside the span the model echoes `x`, `y` and `<think>`; `</think>`
    // follows nothing, so only the forced close closes the span, and after it
    // the model echoes the `m` branches.
    const PROMPT: &str = "</think>mnmompmqmr<think>xyxxy<think>";
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let close = u64::from(serve::MockTokenizer.encode("</think>")[0]);
    let mut drawn_between = false;
    for seed in 1..=8 {
        let mut body = sampled(PROMPT, 40, 0.8, seed);
        body["reasoning_budget"] = json!(9);
        let (want, _) = ids_of(plain, &body);
        drain(&dlog);
        let (got, v) = ids_of(drafted, &body);
        assert_eq!(got, want, "seed {seed}: {v}");
        assert_eq!(
            got.get(9),
            Some(&close),
            "seed {seed}: the forced close: {got:?}"
        );
        let calls = drain(&dlog);
        let first = calls.iter().position(|&c| c == MockCall::Sampled);
        let last = calls.iter().rposition(|&c| c == MockCall::Sampled);
        let (Some(first), Some(last)) = (first, last) else {
            panic!("seed {seed}: no pass: {calls:?}");
        };
        let steps = count(&calls[first..last], MockCall::Next);
        assert!(steps > 0, "seed {seed}: no step between passes: {calls:?}");
        // One of the steps feeds the forced close; any other drew.
        drawn_between |= steps > 1;
    }
    assert!(
        drawn_between,
        "no request drew in a step between two passes"
    );
}

/// A perfect draft: every proposal is the target's argmax and the request
/// takes the argmax (`top_k` 1 under the server's own sampler, still
/// sampled), so every proposed id is kept, `draft_n_accepted` equals
/// `draft_n`, and the ids are the greedy ones.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_perfect_draft_keeps_every_proposal() {
    let dlog = log();
    let mock = DraftMock::new(4096).with_sampled_draft(3, 1, 1);
    let drafted = start_sampling(
        Box::new(mock.logged(Arc::clone(&dlog))),
        1,
        serve::sampling::reference_factory(),
    );
    let body = json!({"prompt": "abcabcabc", "n_predict": 41, "temperature": 0.8, "top_k": 1,
                      "seed": 5});
    let (got, v) = ids_of(drafted, &body);
    let (greedy, _) = ids_of(
        drafted,
        &json!({"prompt": "abcabcabc", "n_predict": 41, "temperature": 0}),
    );
    assert_eq!(got, greedy, "{v}");
    let (n, a) = draft_counts(&v);
    assert!(n > 0, "no proposal: {v}");
    assert_eq!(a, n, "{v}");
    assert!(count(&drain(&dlog), MockCall::Sampled) > 0);
}

/// A greedy request on the drafting mock still takes the greedy passes
/// ([`serve::Engine::advance`]), never a sampled one, with the plain ids and
/// the draft counts of the same mock that declines sampled requests.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_greedy_requests_still_advance() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let declining = serve(drafting(4096).without_sampled(), &plog);
    let body = json!({"prompt": BRANCHY, "n_predict": 40, "temperature": 0});
    let (want, w) = ids_of(declining, &body);
    let (got, v) = ids_of(drafted, &body);
    assert_eq!(got, want, "{v}");
    assert_eq!(draft_counts(&v), draft_counts(&w), "{v} | {w}");
    assert!(draft_counts(&v).0 > 0, "{v}");
    let calls = drain(&dlog);
    assert!(count(&calls, MockCall::Advance) > 0, "{calls:?}");
    assert_eq!(count(&calls, MockCall::Sampled), 0, "{calls:?}");
}

/// What a pass cannot serve steps on the drafting mock: a request that bans
/// the end of generation (`ignore_eos`) and one whose think budget forces its
/// close before a pass's rows fit run no sampled pass and take the plain ids;
/// the same requests without the ban or the budget do pass.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_banned_id_or_a_forced_close_steps() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let close = u64::from(serve::MockTokenizer.encode("</think>")[0]);
    let mut banned = sampled(BRANCHY, 24, 0.8, 3);
    banned["ignore_eos"] = json!(true);
    // A budget of 2 under a pass's four rows: two drawn steps, then the
    // forced close, which ends the request at its `n_predict`.
    let mut budget = sampled("q<think>aa<think>", 3, 0.8, 3);
    budget["reasoning_budget"] = json!(2);
    for (what, body, trigger) in [
        ("ignore_eos", &banned, "ignore_eos"),
        ("reasoning_budget", &budget, "reasoning_budget"),
    ] {
        drain(&dlog);
        let (want, _) = ids_of(plain, body);
        let (got, v) = ids_of(drafted, body);
        assert_eq!(got, want, "{what}: {v}");
        assert_eq!(draft_counts(&v), (0, 0), "{what}: {v}");
        let calls = drain(&dlog);
        assert_eq!(count(&calls, MockCall::Sampled), 0, "{what}: {calls:?}");
        if what == "reasoning_budget" {
            assert_eq!(got.last(), Some(&close), "{got:?}");
        }
        let mut free = body.clone();
        free.as_object_mut().expect("a body").remove(trigger);
        if what == "reasoning_budget" {
            free["n_predict"] = json!(12);
        }
        ids_of(drafted, &free);
        let calls = drain(&dlog);
        assert!(
            count(&calls, MockCall::Sampled) > 0,
            "{what}: without it the request passes: {calls:?}"
        );
    }
}

/// The prompt entries' hold: until it opens, every `prefill` blocks.
#[derive(Default)]
struct Hold {
    state: Mutex<(usize, bool)>,
    cv: Condvar,
}

impl Hold {
    fn enter(&self) {
        let mut g = self.state.lock().expect("the hold");
        g.0 += 1;
        self.cv.notify_all();
        while !g.1 {
            g = self.cv.wait(g).expect("the hold");
        }
    }

    fn reached(&self, n: usize) -> bool {
        let g = self.state.lock().expect("the hold");
        let (g, _) = self
            .cv
            .wait_timeout_while(g, BOUND, |s| s.0 < n)
            .expect("the hold");
        g.0 >= n
    }

    fn open(&self) {
        self.state.lock().expect("the hold").1 = true;
        self.cv.notify_all();
    }
}

/// [`DraftMock`] behind a [`Hold`] in `prefill`, every other call forwarded,
/// the sampled pass and its declaration with them.
struct Held {
    inner: DraftMock,
    hold: Arc<Hold>,
}

impl Engine for Held {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.hold.enter();
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.inner.next(last, out)
    }
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.inner.advance(last, out)
    }
    fn advance_rows(&self) -> usize {
        self.inner.advance_rows()
    }
    fn advance_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, Sampler) {
        self.inner.advance_sampled(last, history, sampler, out)
    }
    fn drafts_sampled(&self) -> bool {
        self.inner.drafts_sampled()
    }
    fn slots(&self) -> usize {
        self.inner.slots()
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.inner.select_slot(slot)
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

/// Waits until `requests_deferred` reaches `n`.
fn wait_deferred(addr: SocketAddr, n: f64) {
    let deadline = Instant::now() + BOUND;
    loop {
        let body = get(addr, "/metrics").body;
        let deferred = body
            .lines()
            .find_map(|l| l.strip_prefix("llamacpp:requests_deferred "))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or_else(|| panic!("no requests_deferred:\n{body}"));
        if deferred >= n {
            return;
        }
        assert!(Instant::now() < deadline, "a request never queued");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Two busy slots: a sampled request that only ever runs beside a greedy one
/// takes plain steps in their shared rounds, no sampled pass, with the plain
/// ids; the same request alone on the same server passes, with those ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_beside_another_busy_slot_a_sampled_request_steps() {
    let (dlog, plog) = (log(), log());
    let hold = Arc::new(Hold::default());
    let held = Held {
        inner: drafting(4096).with_slots(2).logged(Arc::clone(&dlog)),
        hold: Arc::clone(&hold),
    };
    let addr = start_sampling(Box::new(held), 2, penalised());
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let long = json!({"prompt": "abcabcabc", "n_predict": 400, "temperature": 0,
                      "cache_prompt": false, "return_tokens": true});
    let mut short = sampled(BRANCHY, 12, 0.8, 4);
    short["cache_prompt"] = json!(false);
    short["return_tokens"] = json!(true);
    let a = std::thread::spawn(move || post(addr, "/completion", &long));
    assert!(hold.reached(1), "the long prompt never reached the engine");
    let b = {
        let short = short.clone();
        std::thread::spawn(move || post(addr, "/completion", &short))
    };
    wait_deferred(addr, 1.0);
    hold.open();
    let (ra, rb) = (a.join().expect("long"), b.join().expect("short"));
    assert_eq!(
        (ra.status, rb.status),
        (200, 200),
        "{} | {}",
        ra.body,
        rb.body
    );
    let (va, vb) = (ra.json(), rb.json());
    assert!(draft_counts(&va).0 > 0, "the greedy request drafted: {va}");
    assert_eq!(
        draft_counts(&vb),
        (0, 0),
        "the sampled request stepped: {vb}"
    );
    let calls = drain(&dlog);
    assert_eq!(count(&calls, MockCall::Sampled), 0, "{calls:?}");
    let (want, _) = ids_of(plain, &short);
    assert_eq!(vb["tokens"], json!(want), "{vb}");
    let (alone, v) = ids_of(addr, &short);
    assert_eq!(alone, want, "{v}");
    assert!(draft_counts(&v).0 > 0, "alone it passes: {v}");
    assert!(count(&drain(&dlog), MockCall::Sampled) > 0);
}

/// [`DraftMock`] whose sampled pass hands its history back with its last id
/// changed, as long as it was lent.
struct Alters(DraftMock);

impl Engine for Alters {
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
    fn advance_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, Sampler) {
        let (d, mut history, sampler) = self.0.advance_sampled(last, history, sampler, out);
        if let Some(id) = history.last_mut() {
            *id += 1;
        }
        (d, history, sampler)
    }
    fn drafts_sampled(&self) -> bool {
        self.0.drafts_sampled()
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
}

/// A sampled pass whose history comes back other than it was lent, at its
/// length, fails the request by name: the next draw would read ids the
/// request never took.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_history_handed_back_changed_is_a_named_error() {
    let addr = start_sampling(Box::new(Alters(drafting(4096))), 1, penalised());
    let r = post(addr, "/completion", &sampled(BRANCHY, 8, 0.8, 1));
    assert_eq!(r.status, 500, "{}", r.body);
    let msg = r.json()["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert!(msg.contains("handed back another"), "{msg}");
}
