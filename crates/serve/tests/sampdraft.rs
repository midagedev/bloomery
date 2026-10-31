//! Gate: a sampled request on an engine that drafts sampled requests
//! ([`serve::Engine::advance_sampled`]) takes the ids a plain run of the same
//! seed takes, alone in its round or beside another busy slot's pass, and
//! steps where a pass would read what the loop has not settled: a think close
//! it forces. A request that bans an id (`ignore_eos`) or biases ids
//! (`logit_bias`) draws in those passes too, greedy or sampled, from rows
//! whose banned ids are at -inf and whose biased ids carry their bias, and so
//! does a request that asks for probabilities, its values read from the kept
//! rows, or follows a JSON schema, its program's pick taken from each kept
//! row. The drafted mock is
//! [`DraftMock`]; its plain twin is the same mock
//! [`DraftMock::without_sampled`]. Both servers sample with [`penalised`],
//! whose draws read the history, so a pass that hands the sampler another
//! history than a step does takes other ids.

mod common;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{get, post, start_sampling};
use serde_json::{Value, json};
use serve::mock::{Hold, MockCall};
use serve::{
    DraftMock, Drafted, Engine, EngineError, Sampler, SamplerFactory, SamplerRefused,
    SamplingParams, SlotPass, Tokenizer,
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
    Arc::new(
        move |p: &SamplingParams| -> Result<Sampler, SamplerRefused> {
            let mut draw = inner(p)?;
            let mut row: Vec<f32> = Vec::new();
            Ok(Box::new(move |logits: &[f32], history: &[u32]| {
                row.clear();
                row.extend_from_slice(logits);
                for &id in history.iter().rev().take(8) {
                    if let Some(l) = row.get_mut(id as usize) {
                        *l = if *l > 0.0 { *l / 1.3 } else { *l * 1.3 };
                    }
                }
                draw(&row, history)
            }))
        },
    )
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

/// A request that bans the end of generation (`ignore_eos`) drafts as it
/// would without the ban, greedy and sampled: its passes draw from rows
/// whose stop ids are at -inf, and its ids are the plain twin's, none a stop
/// id, to `n_predict`. The prompt ends on an id the mock never saw, so after
/// the first generated id (also unseen) the target's argmax is the end of
/// generation, which the greedy run's first pass proposes: a pass that kept
/// it fails the request.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_banned_id_drafts() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let eos = u64::from(serve::MockTokenizer.eos());
    let (mut proposed, mut accepted) = (0, 0);
    for temperature in [0.0, 0.8] {
        for seed in 1..=4u64 {
            let mut body = sampled("abacadaez", 24, temperature, seed);
            body["ignore_eos"] = json!(true);
            let at = format!("T {temperature}, seed {seed}");
            let (want, w) = ids_of(plain, &body);
            let (got, v) = ids_of(drafted, &body);
            assert_eq!(got, want, "{at}: {v}");
            assert_eq!(got.len(), 24, "{at}: {v}");
            assert!(!got.contains(&eos), "{at}: {got:?}");
            assert_eq!(
                draft_counts(&w),
                (0, 0),
                "{at}: the plain twin drafted: {w}"
            );
            let (n, a) = draft_counts(&v);
            assert!(n > 0, "{at}: the banned request drafted nothing: {v}");
            (proposed, accepted) = (proposed + n, accepted + a);
        }
    }
    assert!(
        0 < accepted && accepted < proposed,
        "the passes kept {accepted} of {proposed} proposed ids: both kept and taken-back rows are \
         needed"
    );
    assert!(count(&drain(&dlog), MockCall::Sampled) > 0, "no pass drew");
    assert_eq!(
        count(&drain(&plog), MockCall::Sampled),
        0,
        "the plain twin ran a pass"
    );
}

/// A request with a `logit_bias` drafts, greedy and sampled, and takes the
/// plain twin's ids: its passes draw from the verified rows with the bias
/// added, as its steps do. -100 on `c` keeps every `c` out, +3 on `d` lifts
/// it over the mock's echo where it follows, so the bias moves the ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_biased_request_drafts() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let id = |c: &str| u64::from(serve::MockTokenizer.encode(c)[0]);
    let bias = json!({id("c").to_string(): -100, id("d").to_string(): 3.0});
    let (mut proposed, mut moved) = (0, false);
    for temperature in [0.0, 0.8] {
        for seed in 1..=4u64 {
            let mut body = sampled(BRANCHY, 32, temperature, seed);
            let (unbiased, _) = ids_of(plain, &body);
            body["logit_bias"] = bias.clone();
            let at = format!("T {temperature}, seed {seed}");
            let (want, w) = ids_of(plain, &body);
            let (got, v) = ids_of(drafted, &body);
            assert_eq!(got, want, "{at}: {v}");
            assert!(!got.contains(&id("c")), "{at}: {got:?}");
            assert_eq!(
                draft_counts(&w),
                (0, 0),
                "{at}: the plain twin drafted: {w}"
            );
            proposed += draft_counts(&v).0;
            moved |= got != unbiased;
        }
    }
    assert!(proposed > 0, "the biased requests drafted nothing");
    assert!(moved, "the bias moved no id");
    assert!(count(&drain(&dlog), MockCall::Sampled) > 0, "no pass drew");
    assert_eq!(
        count(&drain(&plog), MockCall::Sampled),
        0,
        "the plain twin ran a pass"
    );
}

/// A forced think close steps on the drafting mock: a request whose budget
/// forces its close before a pass's rows fit runs no sampled pass and takes
/// the plain ids; the same request without the budget does pass.
// PIN(2026-10-23): hw_a_banned_id_or_a_forced_close_steps keeps its forced
// close arm here; its ignore_eos arm drafts now (hw_a_banned_id_drafts).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_forced_close_steps() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let close = u64::from(serve::MockTokenizer.encode("</think>")[0]);
    // A budget of 2 under a pass's four rows: two drawn steps, then the
    // forced close, which ends the request at its `n_predict`.
    let mut budget = sampled("q<think>aa<think>", 3, 0.8, 3);
    budget["reasoning_budget"] = json!(2);
    drain(&dlog);
    let (want, _) = ids_of(plain, &budget);
    let (got, v) = ids_of(drafted, &budget);
    assert_eq!(got, want, "{v}");
    assert_eq!(draft_counts(&v), (0, 0), "{v}");
    let calls = drain(&dlog);
    assert_eq!(count(&calls, MockCall::Sampled), 0, "{calls:?}");
    assert_eq!(got.last(), Some(&close), "{got:?}");
    let mut free = budget.clone();
    free.as_object_mut()
        .expect("a body")
        .remove("reasoning_budget");
    free["n_predict"] = json!(12);
    ids_of(drafted, &free);
    let calls = drain(&dlog);
    assert!(
        count(&calls, MockCall::Sampled) > 0,
        "without the budget the request passes: {calls:?}"
    );
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

// PIN(2026-10-29): `hw_beside_another_busy_slot_a_sampled_request_steps` became the test
// below: a sampled request's pass rides the round's drafted call beside another busy slot.

/// Two busy slots: a sampled request that only ever runs beside a greedy one
/// passes in their shared rounds, its sampled pass a row of the drafted call
/// of both slots, with the plain ids; the same request alone on the same
/// server passes, with those ids.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_beside_another_busy_slot_a_sampled_request_passes() {
    let (dlog, plog) = (log(), log());
    let hold = Arc::new(Hold::default());
    let held = drafting(4096)
        .with_slots(2)
        .logged(Arc::clone(&dlog))
        .holding(Arc::clone(&hold));
    let addr = start_sampling(Box::new(held), 2, penalised());
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let long = json!({"prompt": "abcabcabc", "n_predict": 400, "temperature": 0,
                      "cache_prompt": false, "return_tokens": true});
    let mut short = sampled(BRANCHY, 12, 0.8, 4);
    short["cache_prompt"] = json!(false);
    short["return_tokens"] = json!(true);
    let a = std::thread::spawn(move || post(addr, "/completion", &long));
    assert!(
        hold.reached(1, BOUND),
        "the long prompt never reached the engine"
    );
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
    assert!(
        draft_counts(&vb).0 > 0,
        "the sampled request drafted beside the greedy one: {vb}"
    );
    let calls = drain(&dlog);
    assert!(count(&calls, MockCall::Sampled) > 0, "{calls:?}");
    assert!(
        calls.iter().any(|c| matches!(c, MockCall::AdvanceSlots(2))),
        "the two slots passed in one call: {calls:?}"
    );
    let (want, _) = ids_of(plain, &short);
    assert_eq!(vb["tokens"], json!(want), "{vb}");
    let (alone, v) = ids_of(addr, &short);
    assert_eq!(alone, want, "{v}");
    assert!(draft_counts(&v).0 > 0, "alone it passes: {v}");
    assert!(count(&drain(&dlog), MockCall::Sampled) > 0);
}

/// A request that asks for probabilities drafts, greedy or sampled, alone
/// and beside another busy slot: its ids and its values are the plain twin's
/// steps', each kept id's value read from its own verified row, and every
/// generated id has its entry.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_an_asking_request_drafts() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let greedy = json!({"prompt": BRANCHY, "n_predict": 40, "temperature": 0, "n_probs": 3});
    let mut drawn = sampled(BRANCHY, 40, 0.8, 5);
    drawn["n_probs"] = json!(3);
    for body in [&greedy, &drawn] {
        let (want, w) = ids_of(plain, body);
        let (got, v) = ids_of(drafted, body);
        assert_eq!(got, want, "{body}: {v}");
        assert_eq!(
            v["completion_probabilities"], w["completion_probabilities"],
            "{body}"
        );
        assert_eq!(
            v["completion_probabilities"].as_array().map(Vec::len),
            Some(got.len()),
            "{body}: one entry a generated id"
        );
        assert!(
            draft_counts(&v).1 > 0,
            "{body}: the passes kept drafted ids: {v}"
        );
        assert_eq!(draft_counts(&w), (0, 0), "{body}: the plain twin drafted");
        assert!(count(&drain(&dlog), MockCall::Sampled) > 0, "{body}");
    }
    let hold = Arc::new(Hold::default());
    let held = drafting(4096)
        .with_slots(2)
        .logged(Arc::clone(&dlog))
        .holding(Arc::clone(&hold));
    let addr = start_sampling(Box::new(held), 2, penalised());
    let long = json!({"prompt": "abcabcabc", "n_predict": 400, "temperature": 0,
                      "cache_prompt": false});
    let mut short = drawn.clone();
    short["cache_prompt"] = json!(false);
    short["return_tokens"] = json!(true);
    let a = std::thread::spawn(move || post(addr, "/completion", &long));
    assert!(
        hold.reached(1, BOUND),
        "the long prompt never reached the engine"
    );
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
    let vb = rb.json();
    let (want, w) = ids_of(plain, &short);
    assert_eq!(vb["tokens"], json!(want), "{vb}");
    assert_eq!(
        vb["completion_probabilities"],
        w["completion_probabilities"]
    );
    assert!(
        draft_counts(&vb).0 > 0,
        "the asking request drafted beside the greedy one: {vb}"
    );
    assert!(
        drain(&dlog)
            .iter()
            .any(|c| matches!(c, MockCall::AdvanceSlots(2))),
        "the two slots passed in one call"
    );
}

/// A request that follows a program drafts: a JSON schema's document on
/// `/completion`, greedy, sampled and asking for probabilities, takes the
/// plain twin's ids and text, its passes picking each kept id from its
/// verified row by the program (a fixed id read from no row, a pick masked to
/// the ids the program allows there), the program moving through the kept
/// ids alone.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_constrained_request_drafts() {
    let (dlog, plog) = (log(), log());
    let drafted = serve(drafting(4096), &dlog);
    let plain = serve(drafting(4096).without_sampled(), &plog);
    let schema = json!({"type": "object", "properties": {
        "id": {"type": "integer", "enum": [7, 8]}, "ok": {"type": "boolean"},
        "tag": {"type": "string", "maxLength": 5}}});
    let rf = json!({"type": "json_schema", "json_schema": {"name": "r", "schema": schema}});
    let greedy = json!({"prompt": BRANCHY, "n_predict": 120, "response_format": rf});
    let mut drawn = sampled(BRANCHY, 120, 0.8, 6);
    drawn["response_format"] = rf;
    let mut asking = greedy.clone();
    asking["n_probs"] = json!(2);
    let mut passed = 0;
    for body in [&greedy, &drawn, &asking] {
        let (want, w) = ids_of(plain, body);
        let (got, v) = ids_of(drafted, body);
        assert_eq!(got, want, "{body}: {v}");
        assert_eq!(v["content"], w["content"], "{body}");
        assert_eq!(
            v["completion_probabilities"], w["completion_probabilities"],
            "{body}"
        );
        let doc: Value = serde_json::from_str(v["content"].as_str().expect("content"))
            .unwrap_or_else(|e| panic!("{e}: {v}"));
        assert!([7, 8].contains(&doc["id"].as_i64().expect("id")), "{doc}");
        assert_eq!(draft_counts(&w), (0, 0), "{body}: the plain twin drafted");
        passed += count(&drain(&dlog), MockCall::Sampled);
    }
    assert!(passed > 0, "no constrained request passed");
    // A pass keeps the fixed `a`, then the program refuses its next row's
    // pick (an infinite penalty leaves no allowed id finite): the two-token
    // answer ends on `a` as the plain twin's does, the refusal past it unread;
    // a longer one ends on the plain twin's refusal once `a` is taken.
    let rf = json!({"type": "json_schema", "json_schema": {"name": "r", "schema": {"enum": ["ab", "ac"]}}});
    let short = json!({"prompt": "\"abbcc", "n_predict": 2, "n_probs": 1, "temperature": 1, "top_k": 1,
        "frequency_penalty": 3e38, "response_format": rf});
    let (want, w) = ids_of(plain, &short);
    let (got, v) = ids_of(drafted, &short);
    assert_eq!((got, &v["content"]), (want, &w["content"]), "{v}");
    assert!(
        count(&drain(&dlog), MockCall::Sampled) > 0,
        "the short request passed: {v}"
    );
    let mut long = short;
    long["n_predict"] = json!(8);
    let ended = |addr| {
        let r = post(addr, "/completion", &long);
        (r.status, r.json()["error"].clone())
    };
    let want = ended(plain);
    assert_ne!(want.0, 200, "{want:?}");
    assert_eq!(ended(drafted), want);
}

/// [`DraftMock`] whose sampled pass hands its history back with its last id
/// changed, as long as it was lent, once its first prompts have completed.
struct Alters(DraftMock, usize);

impl Engine for Alters {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.0.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.0.prefill(ids)?;
        self.1 = self.1.saturating_sub(1);
        Ok(())
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
        if self.1 == 0
            && let Some(id) = history.last_mut()
        {
            *id += 1;
        }
        (d, history, sampler)
    }
    fn drafts_sampled(&self) -> bool {
        self.0.drafts_sampled()
    }
    fn slots(&self) -> usize {
        self.0.slots()
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.0.select_slot(slot)
    }
    fn slot_drafts(&self) -> bool {
        self.0.slot_drafts()
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

/// The default round adapter rejects a same-length history edit and returns
/// the sampler even on that refusal.
#[test]
fn a_round_rejects_a_changed_history_and_restores_its_sampler() {
    let mut engine = Alters(drafting(4096).with_slots(2), 1);
    engine.select_slot(1).expect("the second slot");
    engine.prefill(&[1, 2, 3]).expect("a prompt");
    let history = [1, 2, 3];
    let mut sampler: Option<Sampler> = Some(Box::new(|_, _| 1));
    let mut out = Vec::new();
    let mut rows = [SlotPass {
        slot: 1,
        last: 3,
        out: &mut out,
        drafted: Drafted::default(),
        draw: Some(serve::SlotDraw {
            history: &history,
            sampler: &mut sampler,
        }),
    }];
    let e = engine
        .advance_slots(&mut rows)
        .expect_err("a history edit must fail");
    assert_eq!(e.0, "slot 1: a sampled pass handed back another history");
    assert!(sampler.is_some(), "the sampler comes back on refusal");
}

/// A sampled pass whose history comes back other than it was lent, at its
/// length, fails the request by name: the next draw would read ids the
/// request never took.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_history_handed_back_changed_is_a_named_error() {
    let check = |r: common::Reply| {
        let v = r.json();
        let msg = v["error"]["message"].as_str().expect("a named error");
        assert!(msg.contains("handed back another"), "{msg}");
    };
    let addr = start_sampling(Box::new(Alters(drafting(4096), 1)), 1, penalised());
    let r = post(addr, "/completion", &sampled(BRANCHY, 8, 0.8, 1));
    assert_eq!(r.status, 500, "{}", r.body);
    check(r);

    let hold = Arc::new(Hold::default());
    let held = drafting(4096).with_slots(2).holding(Arc::clone(&hold));
    let addr = start_sampling(Box::new(Alters(held, 2)), 2, penalised());
    let a = std::thread::spawn(move || post(addr, "/completion", &sampled(BRANCHY, 40, 0.8, 1)));
    assert!(
        hold.reached(1, BOUND),
        "the first prompt never reached the engine"
    );
    let b = std::thread::spawn(move || post(addr, "/completion", &sampled(BRANCHY, 40, 0.8, 2)));
    wait_deferred(addr, 1.0);
    hold.open();
    for t in [a, b] {
        let r = t.join().expect("a request");
        assert!(matches!(r.status, 500 | 503), "{}: {}", r.status, r.body);
        check(r);
    }
}

/// [`DraftMock`] of two slots whose round of passes keeps every sampled
/// row's sampler.
struct Keeps(DraftMock);

impl Engine for Keeps {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.0.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.0.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.0.next(last, out)
    }
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.0.advance(last, out)
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
        self.0.advance_sampled(last, history, sampler, out)
    }
    fn drafts_sampled(&self) -> bool {
        self.0.drafts_sampled()
    }
    fn slots(&self) -> usize {
        self.0.slots()
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.0.select_slot(slot)
    }
    fn slot_drafts(&self) -> bool {
        self.0.slot_drafts()
    }
    fn advance_slots(&mut self, rows: &mut [SlotPass<'_>]) -> Result<(), EngineError> {
        self.0.advance_slots(rows)?;
        for d in rows.iter_mut().filter_map(|r| r.draw.as_mut()) {
            *d.sampler = None;
        }
        Ok(())
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

/// A round of passes that keeps a sampled row's sampler fails its requests
/// by name: the request's next draw would have no sampler.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_sampler_kept_by_a_round_is_a_named_error() {
    let hold = Arc::new(Hold::default());
    let held = drafting(4096).with_slots(2).holding(Arc::clone(&hold));
    let addr = start_sampling(Box::new(Keeps(held)), 2, penalised());
    let a = std::thread::spawn(move || post(addr, "/completion", &sampled(BRANCHY, 40, 0.8, 1)));
    assert!(
        hold.reached(1, BOUND),
        "the first prompt never reached the engine"
    );
    let b = std::thread::spawn(move || post(addr, "/completion", &sampled(BRANCHY, 40, 0.8, 2)));
    wait_deferred(addr, 1.0);
    hold.open();
    let msgs: Vec<String> = [a, b]
        .map(|t| {
            let r = t.join().expect("a request");
            assert_eq!(r.status, 500, "{}", r.body);
            r.json()["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .to_owned()
        })
        .into();
    assert!(
        msgs.iter()
            .all(|m| m.contains("kept the request's sampler")),
        "{msgs:?}"
    );
}
