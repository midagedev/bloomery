//! Gate: the HTTP server over the mock engine — JSON shapes, stream framing and
//! stop rules against llama-server's field names (`tests/fixtures`).
//!
//! The mock's greedy output is derived by hand from its bigram rule (see
//! `serve::mock`): for the chat below the rendered prompt ends in `</think>`, whose
//! earlier occurrence is followed by `abc<｜end▁of▁sentence｜>`, so greedy decoding
//! yields `a`, `b`, `c`, EOS.

mod common;

use common::{assert_keys, call, fixture_keys, get, post, start};
use serde_json::{Value, json};

fn chat_body(extra: Value) -> Value {
    let mut b = json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "abc"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0,
    });
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e);
    }
    b
}

fn assert_timings(t: &Value, prompt_n: u64, predicted_n: u64) {
    assert_keys(t, "timings");
    assert_eq!(t["prompt_n"], prompt_n, "{t}");
    assert_eq!(t["predicted_n"], predicted_n, "{t}");
    assert!(t["prompt_ms"].as_f64().is_some_and(|x| x >= 0.0), "{t}");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_non_stream_shape() {
    let addr = start(4096);
    let r = post(addr, "/v1/chat/completions", &chat_body(json!({})));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(v["object"], "chat.completion");
    assert!(
        v["id"].as_str().is_some_and(|s| s.starts_with("chatcmpl-")),
        "{v}"
    );
    assert!(v["created"].is_u64());
    assert_eq!(v["model"], "m");
    let c = &v["choices"][0];
    assert_eq!(c["index"], 0);
    assert_eq!(c["message"]["role"], "assistant");
    assert_eq!(c["message"]["content"], "abc");
    assert_eq!(c["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 15);
    assert_eq!(v["usage"]["completion_tokens"], 4);
    assert_eq!(v["usage"]["total_tokens"], 19);
    assert_timings(&v["timings"], 15, 4);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_stream_framing_matches_non_stream() {
    let addr = start(4096);
    // `a` has four followers in this history, so a sampler at T = 1.5 branches;
    // min_p 0.01 keeps only those followers (the rest sit 12 logits down).
    let branching = |extra: Value| {
        let mut b = chat_body(extra);
        b["messages"][1]["content"] = json!("abacadae");
        b
    };
    let sampled = json!({"temperature": 1.5, "top_k": 0, "top_p": 1.0, "min_p": 0.01, "seed": 42, "max_tokens": 24});
    let content_for = |seed: u64| {
        let mut b = sampled.clone();
        b["seed"] = json!(seed);
        post(addr, "/v1/chat/completions", &branching(b)).json()["choices"][0]["message"]["content"]
            .as_str()
            .expect("content")
            .to_owned()
    };
    let by_seed: Vec<String> = (1..=5).map(content_for).collect();
    assert!(
        by_seed.iter().any(|c| c != &by_seed[0]),
        "the sampler never branched: {by_seed:?}"
    );
    assert_eq!(content_for(42), content_for(42), "same seed, same content");
    let whole = post(addr, "/v1/chat/completions", &branching(sampled.clone())).json();
    let content = whole["choices"][0]["message"]["content"]
        .as_str()
        .expect("content")
        .to_owned();

    let mut streamed = sampled;
    streamed["stream"] = json!(true);
    streamed["stream_options"] = json!({"include_usage": true});
    let r = post(addr, "/v1/chat/completions", &branching(streamed));
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(
        r.header("content-type")
            .is_some_and(|c| c.starts_with("text/event-stream"))
    );
    let ev = r.events();
    assert_eq!(
        ev.last().map(String::as_str),
        Some("[DONE]"),
        "stream must end with [DONE]: {ev:?}"
    );
    let chunks: Vec<Value> = ev[..ev.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).expect("chunk JSON"))
        .collect();
    assert!(
        chunks
            .iter()
            .all(|c| c["object"] == "chat.completion.chunk")
    );
    let first = &chunks[0]["choices"][0];
    assert_eq!(first["delta"]["role"], "assistant");
    assert!(first["delta"]["content"].is_null());
    let usage = chunks.last().expect("usage chunk");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(
        usage["usage"]["completion_tokens"],
        whole["usage"]["completion_tokens"]
    );
    assert_keys(&usage["timings"], "timings");
    let fin = &chunks[chunks.len() - 2]["choices"][0];
    assert_eq!(fin["delta"], json!({}));
    assert_eq!(fin["finish_reason"], whole["choices"][0]["finish_reason"]);
    let deltas: String = chunks[1..chunks.len() - 2]
        .iter()
        .map(|c| {
            assert!(c["choices"][0]["finish_reason"].is_null(), "{c}");
            c["choices"][0]["delta"]["content"]
                .as_str()
                .expect("delta")
                .to_owned()
        })
        .collect();
    assert_eq!(
        deltas, content,
        "concatenated deltas vs the non-stream content, same seed"
    );
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_max_tokens_is_length() {
    let addr = start(4096);
    let v = post(
        addr,
        "/v1/chat/completions",
        &chat_body(json!({"max_tokens": 2})),
    )
    .json();
    assert_eq!(v["choices"][0]["message"]["content"], "ab");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_stop_string_is_cut() {
    let addr = start(4096);
    let v = post(
        addr,
        "/v1/chat/completions",
        &chat_body(json!({"stop": ["bc"]})),
    )
    .json();
    assert_eq!(v["choices"][0]["message"]["content"], "a");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // The held-back "b" must not leak into a stream either.
    let r = post(
        addr,
        "/v1/chat/completions",
        &chat_body(json!({"stop": "bc", "stream": true})),
    );
    let text: String = r
        .events()
        .iter()
        .filter(|e| *e != "[DONE]")
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter_map(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(text, "a");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_errors_are_openai_objects() {
    let addr = start(64);
    let check = |r: common::Reply, code: u16, kind: &str| {
        assert_eq!(r.status, code, "{}", r.body);
        let e = &r.json()["error"];
        assert_eq!(e["code"], code, "{e}");
        assert_eq!(e["type"], kind, "{e}");
        assert!(e["message"].as_str().is_some_and(|m| !m.is_empty()), "{e}");
    };
    check(
        call(addr, "POST", "/v1/chat/completions", Some("{not json")),
        400,
        "invalid_request_error",
    );
    check(
        post(addr, "/v1/chat/completions", &json!({"model": "m"})),
        400,
        "invalid_request_error",
    );
    check(
        post(addr, "/completion", &json!({"prompt": "ab", "n_probs": 3})),
        400,
        "invalid_request_error",
    );
    check(
        post(addr, "/completion", &json!({"prompt": 7})),
        400,
        "invalid_request_error",
    );
    check(
        post(addr, "/completion", &json!({"prompt": "x".repeat(64)})),
        400,
        "exceed_context_size_error",
    );
    check(get(addr, "/nope"), 404, "not_found_error");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_completion_non_stream_fields() {
    let addr = start(4096);
    let v = post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
    )
    .json();
    assert_keys(&v, "completion_final");
    assert_keys(&v["generation_settings"], "generation_settings");
    assert_eq!(v["content"], "abcab");
    assert_eq!(v["stop"], true);
    assert_eq!(v["tokens_predicted"], 5);
    assert_eq!(v["tokens_evaluated"], 6);
    assert_eq!(v["stopped_limit"], true);
    assert_eq!(v["stopped_eos"], false);
    assert_eq!(v["stop_type"], "limit");
    assert_eq!(v["prompt"], "abcabc");
    assert_timings(&v["timings"], 6, 5);

    // The same prompt as token ids (mock byte ids are 6 + byte).
    let ids: Vec<u32> = "abcabc".bytes().map(|b| 6 + u32::from(b)).collect();
    let by_ids = post(
        addr,
        "/completion",
        &json!({"prompt": ids, "n_predict": 5, "temperature": 0}),
    )
    .json();
    assert_eq!(by_ids["content"], "abcab");

    let eos = post(
        addr,
        "/completion",
        &json!({"prompt": "xyz", "temperature": 0}),
    )
    .json();
    assert_eq!(eos["content"], "");
    assert_eq!(eos["stopped_eos"], true);
    assert_eq!(eos["tokens_predicted"], 1);

    let word = post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "stop": ["ca"], "temperature": 0}),
    )
    .json();
    assert_eq!(word["content"], "ab");
    assert_eq!(word["stopped_word"], true);
    assert_eq!(word["stopping_word"], "ca");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_completion_stream_framing() {
    let addr = start(4096);
    let body = json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0, "stream": true});
    let r = post(addr, "/completion", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let ev = r.events();
    assert!(
        !ev.iter().any(|e| e == "[DONE]"),
        "/completion streams carry no [DONE] (llama-server)"
    );
    let chunks: Vec<Value> = ev
        .iter()
        .map(|e| serde_json::from_str(e).expect("chunk JSON"))
        .collect();
    let (last, parts) = chunks.split_last().expect("chunks");
    let mut text = String::new();
    for c in parts {
        assert_keys(c, "completion_partial");
        assert_eq!(c["stop"], false, "{c}");
        text.push_str(c["content"].as_str().expect("content"));
    }
    assert_eq!(text, "abcab");
    assert_keys(last, "completion_final");
    assert_eq!(last["stop"], true);
    assert_eq!(last["content"], "");
    assert_timings(&last["timings"], 6, 5);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tokenize_detokenize_round_trip() {
    let addr = start(4096);
    let text = "héllo <｜User｜>!";
    let t = post(addr, "/tokenize", &json!({"content": text})).json();
    let tokens = t["tokens"].clone();
    assert!(
        tokens.as_array().is_some_and(|a| a.contains(&json!(2))),
        "the special string is one token: {t}"
    );
    let d = post(addr, "/detokenize", &json!({"tokens": tokens})).json();
    assert_eq!(d["content"], text);
    let p = post(
        addr,
        "/tokenize",
        &json!({"content": "ab", "with_pieces": true}),
    )
    .json();
    assert_eq!(
        p["tokens"],
        json!([{"id": 103, "piece": "a"}, {"id": 104, "piece": "b"}])
    );
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_props_slots_metrics_health_models() {
    let addr = start(4096);
    let h = get(addr, "/health");
    assert_eq!((h.status, h.json()["status"].clone()), (200, json!("ok")));

    let props = get(addr, "/props").json();
    assert_keys(&props, "props");
    assert_eq!(props["total_slots"], 1);
    assert_eq!(props["chat_template"], common::V41_TEMPLATE);
    assert_eq!(props["default_generation_settings"]["n_ctx"], 4096);
    assert_keys(&props["default_generation_settings"], "generation_settings");

    let m = get(addr, "/v1/models").json();
    assert_eq!(m["object"], "list");
    assert_eq!(m["data"][0]["id"], "mock");

    post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
    );
    let slots = get(addr, "/slots").json();
    let s = &slots[0];
    assert_keys(s, "slot");
    assert_keys(&s["next_token"], "slot_next_token");
    assert_eq!(s["is_processing"], false);
    assert_eq!(s["n_past"], 10);

    let r = get(addr, "/metrics");
    assert!(
        r.header("content-type")
            .is_some_and(|c| c.starts_with("text/plain; version=0.0.4"))
    );
    for name in fixture_keys("metrics") {
        let full = format!("llamacpp:{name}");
        assert!(
            r.body.contains(&format!("# HELP {full} ")),
            "no HELP for {full}"
        );
        assert!(
            r.body.contains(&format!("# TYPE {full} ")),
            "no TYPE for {full}"
        );
        assert!(
            r.body.lines().any(|l| l
                .split_once(' ')
                .is_some_and(|(k, v)| k == full && v.parse::<f64>().is_ok())),
            "no sample for {full}:\n{}",
            r.body
        );
    }
    let line = |k: &str| {
        r.body
            .lines()
            .find_map(|l| l.strip_prefix(&format!("llamacpp:{k} ")))
            .and_then(|v| v.parse::<f64>().ok())
            .expect("metric value")
    };
    assert_eq!(line("prompt_tokens_total"), 6.0);
    assert_eq!(line("tokens_predicted_total"), 5.0);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_two_requests_share_the_slot() {
    let addr = start(4096);
    let workers: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(move || {
                post(addr, "/v1/chat/completions", &chat_body(json!({}))).json()
            })
        })
        .collect();
    for w in workers {
        let v = w.join().expect("worker");
        assert_eq!(v["choices"][0]["message"]["content"], "abc");
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_unsupported_fields_are_refused() {
    let addr = start(4096);
    let refused = [
        ("response_format", json!({"type": "json_object"})),
        ("json_schema", json!({"type": "object"})),
        ("grammar", json!("root ::= \"a\"")),
        ("logprobs", json!(true)),
        ("top_logprobs", json!(2)),
        ("n", json!(2)),
        ("tool_choice", json!("required")),
    ];
    for (field, value) in refused {
        for path in ["/completion", "/v1/chat/completions"] {
            let mut b = chat_body(json!({}));
            b["prompt"] = json!("ab");
            b[field] = value.clone();
            let r = post(addr, path, &b);
            assert_eq!(r.status, 400, "{path} {field}: {}", r.body);
            let e = &r.json()["error"];
            assert_eq!(e["type"], "invalid_request_error", "{path} {field}: {e}");
            assert!(
                e["message"].as_str().is_some_and(|m| m.contains(field)),
                "{path} {field}: the message must name the field: {e}"
            );
        }
    }
    let neutral = [
        ("response_format", json!({"type": "text"})),
        ("json_schema", Value::Null),
        ("grammar", json!("")),
        ("logprobs", json!(false)),
        ("top_logprobs", json!(0)),
        ("n", json!(1)),
        ("tools", json!([])),
        (
            "tools",
            json!([{"type": "function", "function": {"name": "f"}}]),
        ),
        ("tool_choice", json!("none")),
        ("tool_choice", json!("auto")),
    ];
    for (field, value) in neutral {
        let mut b = chat_body(json!({}));
        b[field] = value;
        let r = post(addr, "/v1/chat/completions", &b);
        assert_eq!(r.status, 200, "{field}: {}", r.body);
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_models_created_is_process_start() {
    let addr = start(4096);
    // Cross a second boundary so a per-call clock cannot equal the start time.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let start_unix: u64 = get(addr, "/metrics")
        .header("Process-Start-Time-Unix")
        .and_then(|v| v.parse().ok())
        .expect("Process-Start-Time-Unix header");
    let a = get(addr, "/v1/models").json()["data"][0]["created"].clone();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let b = get(addr, "/v1/models").json()["data"][0]["created"].clone();
    assert_eq!(a, b, "two calls, same created");
    assert_eq!(a, start_unix, "created is the process start");
}

fn metric(body: &str, k: &str) -> f64 {
    body.lines()
        .find_map(|l| l.strip_prefix(&format!("llamacpp:{k} ")))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no value for {k}:\n{body}"))
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_metrics_throughput_survives_scrapes() {
    let addr = start(4096);
    post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
    );
    let first = get(addr, "/metrics").body;
    let second = get(addr, "/metrics").body;
    for k in ["prompt_tokens_seconds", "predicted_tokens_seconds"] {
        let (a, b) = (metric(&first, k), metric(&second, k));
        assert!(a > 0.0 && a.is_finite(), "{k} first scrape: {a}");
        assert_eq!(a, b, "{k}: a scrape must not reset the gauge");
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_metrics_kv_cache_usage_ratio() {
    let ctx = 4096;
    let addr = start(ctx);
    let idle = get(addr, "/metrics").body;
    assert_eq!(metric(&idle, "kv_cache_usage_ratio"), 0.0);
    // Six prompt tokens and five generated: the mock's cache holds ten positions
    // (the last generated token is never evaluated).
    post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
    );
    let body = get(addr, "/metrics").body;
    assert_eq!(metric(&body, "kv_cache_tokens"), 10.0);
    assert_eq!(
        metric(&body, "kv_cache_usage_ratio"),
        10.0 / ctx as f64,
        "n_past / ctx_max"
    );
}

/// The mock engine with a latch in `next`: its `at`-th `next` and every later one
/// block until `release`, and the first of them says so through `entered`.
struct Held {
    inner: serve::MockEngine,
    latch: std::sync::Arc<Latch>,
    at: usize,
    calls: usize,
}

impl Held {
    fn new(ctx: usize, at: usize, latch: std::sync::Arc<Latch>) -> Held {
        Held {
            inner: serve::MockEngine::new(ctx),
            latch,
            at,
            calls: 0,
        }
    }
}

#[derive(Default)]
struct Latch {
    state: std::sync::Mutex<(bool, bool)>,
    cv: std::sync::Condvar,
}

impl Latch {
    fn wait_entered(&self, bound: std::time::Duration) -> bool {
        let g = self.state.lock().expect("latch");
        let (g, _) = self
            .cv
            .wait_timeout_while(g, bound, |s| !s.0)
            .expect("latch");
        g.0
    }

    fn release(&self) {
        self.state.lock().expect("latch").1 = true;
        self.cv.notify_all();
    }
}

impl serve::Engine for Held {
    fn tokenizer(&self) -> std::sync::Arc<dyn serve::Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), serve::EngineError> {
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, serve::EngineError> {
        self.calls += 1;
        if self.calls >= self.at {
            let mut g = self.latch.state.lock().expect("latch");
            g.0 = true;
            self.latch.cv.notify_all();
            while !g.1 {
                g = self.latch.cv.wait(g).expect("latch");
            }
        }
        self.inner.next(last, out)
    }
    fn reset(&mut self) -> Result<(), serve::EngineError> {
        self.inner.reset()
    }
    fn ctx_max(&self) -> usize {
        self.inner.ctx_max()
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn save_state(
        &self,
        out: &mut dyn std::io::Write,
    ) -> Result<serve::SavedState, serve::StateError> {
        self.inner.save_state(out)
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tokenize_answers_while_a_generation_holds_the_engine() {
    let latch = std::sync::Arc::new(Latch::default());
    let engine = Held::new(4096, 1, latch.clone());
    let addr = common::start_with(Box::new(engine));
    let gen_thread =
        std::thread::spawn(move || post(addr, "/v1/chat/completions", &chat_body(json!({}))));
    assert!(
        latch.wait_entered(std::time::Duration::from_secs(10)),
        "the generation never reached the engine"
    );
    assert_eq!(get(addr, "/health").json()["slots_processing"], 1);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(post(addr, "/tokenize", &json!({"content": "<｜User｜>hi"})));
    });
    let r = rx.recv_timeout(std::time::Duration::from_secs(5));
    latch.release();
    let r = r.expect("/tokenize waited on the engine lock while a generation held it");
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["tokens"], json!([2, 110, 111]));
    let done = gen_thread.join().expect("generation thread");
    assert_eq!(done.json()["choices"][0]["message"]["content"], "abc");
}

/// A spawned server, killed and reaped if the test panics before it exits.
struct Reaped(std::process::Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

/// Starts `bloomery-serve --mock-fail-at k` on a free port; returns the child,
/// its address and a channel of its stderr lines.
fn spawn_failing(
    k: usize,
) -> (
    Reaped,
    std::net::SocketAddr,
    std::sync::mpsc::Receiver<String>,
) {
    spawn_mock(&["--port", "0", "--mock-fail-at", &k.to_string()])
}

/// Starts `bloomery-serve <args>` (which must bind port 0); returns the child,
/// its address and a channel of its stderr lines.
fn spawn_mock(
    args: &[&str],
) -> (
    Reaped,
    std::net::SocketAddr,
    std::sync::mpsc::Receiver<String>,
) {
    use std::io::BufRead;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_bloomery-serve"))
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn bloomery-serve");
    let err = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                return;
            }
        }
    });
    let first = rx
        .recv_timeout(std::time::Duration::from_secs(20))
        .expect("bloomery-serve printed no address");
    let addr = first
        .rsplit_once("http://")
        .and_then(|(_, a)| a.parse().ok())
        .unwrap_or_else(|| panic!("no address in {first:?}"));
    (Reaped(child), addr, rx)
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_engine_error_is_500_then_503_then_exit() {
    // Next #1 answers the prompt's last token; #2, the first generated token's
    // step, fails.
    let (mut served, addr, stderr) = spawn_failing(2);
    let child = &mut served.0;
    let r = post(
        addr,
        "/completion",
        &json!({"prompt": "abcab", "n_predict": 8, "temperature": 0}),
    );
    assert_eq!(r.status, 500, "{}", r.body);
    let msg = r.json()["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert!(msg.contains("injected failure at next #2"), "{msg}");
    let h = get(addr, "/health");
    assert_eq!(h.status, 503, "{}", h.body);
    assert_eq!(h.json()["status"], "error");
    assert!(
        h.json()["reason"]
            .as_str()
            .is_some_and(|s| s.contains("injected failure")),
        "{}",
        h.body
    );
    let again = post(
        addr,
        "/completion",
        &json!({"prompt": "ab", "n_predict": 1}),
    );
    assert_eq!(again.status, 503, "{}", again.body);
    let t0 = std::time::Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().expect("try_wait") {
            break s;
        }
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(20),
            "bloomery-serve kept serving a failed engine for 20 s"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(70), "{status:?}");
    // The reader thread ends at the child's EOF, so this ends too.
    let block: Vec<String> = stderr.iter().collect();
    let block = block.join("\n");
    assert!(block.contains("the engine failed"), "{block}");
    assert!(block.contains("  engine: mock position=5"), "{block}");
    assert!(
        block.contains("  error: engine: mock: injected failure at next #2"),
        "{block}"
    );
}

/// `/props`' `engine` object (toktape's shape): the server's name, its version
/// marked as the mock's, the process argv verbatim and its pid; the mock has no
/// model file, placement or draft, so those keys are absent, not null.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_props_engine_object() {
    let args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--ctx-size",
        "2048",
        "--alias",
        "two words",
    ];
    let (served, addr, _stderr) = spawn_mock(&args);
    let props = get(addr, "/props").json();
    let e = &props["engine"];
    assert_eq!(e["name"], "bloomery", "{props}");
    let version = e["version"].as_str().unwrap_or_default();
    let commit = version
        .strip_prefix(concat!(env!("CARGO_PKG_VERSION"), " ("))
        .and_then(|v| v.strip_suffix(") mock"))
        .unwrap_or_default();
    assert!(!commit.is_empty(), "version {version:?}");
    let mut argv = vec![env!("CARGO_BIN_EXE_bloomery-serve").to_owned()];
    argv.extend(args.iter().map(|a| (*a).to_owned()));
    assert_eq!(e["args"], json!(argv), "{e}");
    assert_eq!(e["server_pid"], served.0.id(), "{e}");
    for key in ["model", "placement", "draft"] {
        assert!(e.get(key).is_none(), "the mock reports no {key}: {e}");
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_completion_return_tokens() {
    let addr = start(4096);
    let body = |extra: Value| {
        let mut b = json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0});
        if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
            b.extend(e);
        }
        b
    };
    let v = post(addr, "/completion", &body(json!({"return_tokens": true}))).json();
    // "a b c a b" in the mock's byte ids (six specials, then byte + 6).
    assert_eq!(v["tokens"], json!([103, 104, 105, 103, 104]), "{v}");
    assert_eq!(v["content"], "abcab");
    let v = post(addr, "/completion", &body(json!({}))).json();
    assert!(v.get("tokens").is_none(), "{v}");
}

/// Feeds `ids` from the engine's position (`prefill` of all but the last, `next`
/// of the last) and returns `count` greedy ids.
fn greedy(e: &mut dyn serve::Engine, ids: &[u32], count: usize) -> Vec<u32> {
    let (last, rest) = ids.split_last().expect("ids to feed");
    e.prefill(rest).expect("prefill");
    let mut tok = e.next(*last, None).expect("next");
    let mut out = vec![tok];
    while out.len() < count {
        tok = e.next(tok, None).expect("next");
        out.push(tok);
    }
    out
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_cut_then_prefill_equals_one_prefill() {
    use serve::{Engine, MockEngine, MockTokenizer, Tokenizer};
    let t = MockTokenizer;
    let prompt = t.encode("abcab");
    let k = 3;
    let mut one = MockEngine::new(64);
    let want = greedy(&mut one, &prompt, 6);
    // Past `k` the cache holds a different tail; without the cut it changes the output.
    let mut over = prompt[..k].to_vec();
    over.extend(t.encode("bz"));
    let mut stale = MockEngine::new(64);
    stale.prefill(&over).expect("prefill");
    assert_ne!(
        greedy(&mut stale, &prompt[k..], 6),
        want,
        "the stale tail must matter, or this gate proves nothing"
    );
    let mut split = MockEngine::new(64);
    split.prefill(&over).expect("prefill");
    assert_eq!(split.keepable(k), k);
    split.cut(k).expect("cut");
    assert_eq!(
        greedy(&mut split, &prompt[k..], 6),
        want,
        "prefill(P[..k] + tail); cut(k); prefill(P[k..]) against prefill(P)"
    );
}

/// The mock engine without `cut`: the trait's defaults.
struct NoCut(serve::MockEngine);

impl serve::Engine for NoCut {
    fn tokenizer(&self) -> std::sync::Arc<dyn serve::Tokenizer> {
        self.0.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), serve::EngineError> {
        self.0.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, serve::EngineError> {
        self.0.next(last, out)
    }
    fn reset(&mut self) -> Result<(), serve::EngineError> {
        self.0.reset()
    }
    fn ctx_max(&self) -> usize {
        self.0.ctx_max()
    }
    fn describe(&self) -> String {
        self.0.describe()
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_cache_prompt_reuses_the_common_prefix() {
    // A leaves "abcabc" + "abca" in the cache (its last generated id is never fed);
    // B shares "abcabcab" with it.
    let a = json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0});
    let b = |extra: Value| {
        let mut v =
            json!({"prompt": "abcabcabb", "n_predict": 4, "temperature": 0, "return_tokens": true});
        if let (Value::Object(v), Value::Object(e)) = (&mut v, extra) {
            v.extend(e);
        }
        v
    };
    let fresh = post(start(4096), "/completion", &b(json!({}))).json();
    assert_eq!(fresh["timings"]["cache_n"], 0, "{fresh}");

    let addr = start(4096);
    post(addr, "/completion", &a);
    let warm = post(addr, "/completion", &b(json!({}))).json();
    assert_eq!(warm["timings"]["cache_n"], 8, "{warm}");
    // llama-server's counts: `prompt_n` is the ids evaluated, `tokens_evaluated`
    // the whole prompt.
    assert_eq!(warm["timings"]["prompt_n"], 1, "{warm}");
    assert_eq!(warm["tokens_evaluated"], 9, "{warm}");
    assert_eq!(
        warm["tokens"], fresh["tokens"],
        "warm {warm}\nfresh {fresh}"
    );
    assert_eq!(warm["content"], fresh["content"]);

    post(addr, "/completion", &a);
    let off = post(addr, "/completion", &b(json!({"cache_prompt": false}))).json();
    assert_eq!(off["timings"]["cache_n"], 0, "{off}");
    assert_eq!(off["tokens"], fresh["tokens"]);

    // An engine without `cut` resets every request instead of failing.
    let plain = common::start_with(Box::new(NoCut(serve::MockEngine::new(4096))));
    post(plain, "/completion", &a);
    let r = post(plain, "/completion", &b(json!({})));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["timings"]["cache_n"], 0);
    assert_eq!(r.json()["tokens"], fresh["tokens"]);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ignore_eos_does_not_outlive_its_request() {
    let addr = start(4096);
    // "z" was never seen, so the mock predicts EOS at once; ignore_eos runs on.
    let held = post(
        addr,
        "/completion",
        &json!({"prompt": "xyz", "ignore_eos": true, "n_predict": 4, "temperature": 0}),
    )
    .json();
    assert_eq!(held["tokens_predicted"], 4, "{held}");
    assert_eq!(held["stopped_limit"], true, "{held}");
    let plain = post(
        addr,
        "/completion",
        &json!({"prompt": "xyz", "temperature": 0}),
    )
    .json();
    assert_eq!(plain["generation_settings"]["ignore_eos"], false, "{plain}");
    assert_eq!(plain["stopped_eos"], true, "{plain}");
    assert_eq!(plain["tokens_predicted"], 1, "{plain}");
    assert_eq!(plain["content"], "", "{plain}");
    assert_eq!(plain["timings"]["cache_n"], 2, "{plain}");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_streams_report_cache_n() {
    let addr = start(4096);
    post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
    );
    let r = post(
        addr,
        "/completion",
        &json!({"prompt": "abcabcabb", "n_predict": 4, "temperature": 0, "stream": true, "return_progress": true}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let chunks: Vec<Value> = r
        .events()
        .iter()
        .map(|e| serde_json::from_str(e).expect("chunk JSON"))
        .collect();
    let progress = chunks
        .iter()
        .find_map(|c| c.get("prompt_progress"))
        .expect("a prompt_progress chunk");
    assert_eq!(progress["cache"], 8, "{progress}");
    assert_eq!(progress["total"], 9, "{progress}");
    assert_eq!(progress["processed"], 9, "{progress}");
    let last = chunks.last().expect("chunks");
    assert_eq!(last["timings"]["cache_n"], 8, "{last}");
    assert_eq!(last["timings"]["prompt_n"], 1, "{last}");

    // The same chat twice: the second keeps all of its prompt but the last token
    // (15 ids; the first left them and "abc" in the cache).
    let mut streamed = chat_body(json!({"stream": true}));
    post(addr, "/v1/chat/completions", &streamed);
    streamed["stream_options"] = json!({"include_usage": true});
    let r = post(addr, "/v1/chat/completions", &streamed);
    let ev = r.events();
    let last: Value = serde_json::from_str(&ev[ev.len() - 2]).expect("usage chunk");
    assert_eq!(last["timings"]["cache_n"], 14, "{last}");
    assert_eq!(last["timings"]["prompt_n"], 1, "{last}");
    assert_eq!(last["usage"]["prompt_tokens"], 15, "{last}");
    assert_eq!(
        last["usage"]["prompt_tokens_details"]["cached_tokens"], 14,
        "{last}"
    );
    assert_eq!(last["usage"]["total_tokens"], 19, "{last}");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_same_prompt_keeps_all_but_its_last_id() {
    let addr = start(4096);
    let body = json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0, "return_tokens": true});
    let first = post(addr, "/completion", &body).json();
    assert_eq!(first["timings"]["cache_n"], 0, "{first}");
    assert_eq!(first["timings"]["prompt_n"], 6, "{first}");
    // The cache holds the prompt and four generated ids; the prompt's last id
    // is evaluated again for its logits.
    let again = post(addr, "/completion", &body).json();
    assert_eq!(again["timings"]["cache_n"], 5, "{again}");
    assert_eq!(again["timings"]["prompt_n"], 1, "{again}");
    assert_eq!(again["tokens_evaluated"], 6, "{again}");
    assert_eq!(again["tokens"], first["tokens"], "{again}\n{first}");
}

/// The mock engine under a cut rule like V4.1's (`Body::keep_point` with
/// ratios 2 and 1): all positions from `held − 1` on, else an even count.
struct Rounding {
    inner: serve::MockEngine,
    held: usize,
}

impl serve::Engine for Rounding {
    fn tokenizer(&self) -> std::sync::Arc<dyn serve::Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), serve::EngineError> {
        self.inner.prefill(ids)?;
        self.held += ids.len();
        Ok(())
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, serve::EngineError> {
        let g = self.inner.next(last, out)?;
        self.held += 1;
        Ok(g)
    }
    fn reset(&mut self) -> Result<(), serve::EngineError> {
        self.held = 0;
        self.inner.reset()
    }
    fn keepable(&self, n: usize) -> usize {
        let n = n.min(self.held);
        if n + 1 >= self.held { n } else { n - n % 2 }
    }
    fn cut(&mut self, n: usize) -> Result<(), serve::EngineError> {
        if self.keepable(n) != n {
            return Err(serve::EngineError(format!(
                "cut to {n} of {}: not a kept point",
                self.held
            )));
        }
        self.held = n;
        self.inner.cut(n)
    }
    fn ctx_max(&self) -> usize {
        self.inner.ctx_max()
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_cache_n_is_what_the_engine_keeps() {
    // A leaves "abcabc" + "abca" (10 ids); B shares its first 7 and is 8 long.
    let a = json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0});
    let b = json!({"prompt": "abcabcaz", "n_predict": 4, "temperature": 0, "return_tokens": true});
    let fresh = post(start(4096), "/completion", &b).json();
    let rounding = || {
        common::start_with(Box::new(Rounding {
            inner: serve::MockEngine::new(4096),
            held: 0,
        }))
    };
    let addr = rounding();
    post(addr, "/completion", &a);
    let warm = post(addr, "/completion", &b).json();
    assert_eq!(warm["timings"]["cache_n"], 6, "{warm}");
    assert_eq!(warm["timings"]["prompt_n"], 2, "{warm}");
    assert_eq!(warm["tokens"], fresh["tokens"], "{warm}\n{fresh}");
    // The same engine keeps an odd count where it is all but the last held id.
    let c =
        json!({"prompt": "abcabcabca", "n_predict": 2, "temperature": 0, "return_tokens": true});
    let addr = rounding();
    post(addr, "/completion", &a);
    let tail = post(addr, "/completion", &c).json();
    assert_eq!(tail["timings"]["cache_n"], 9, "{tail}");
    assert_eq!(
        tail["tokens"],
        post(start(4096), "/completion", &c).json()["tokens"]
    );
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_non_integer_number_is_refused() {
    let addr = start(4096);
    let r = post(
        addr,
        "/completion",
        &json!({"prompt": "ab", "n_predict": 2.5}),
    );
    assert_eq!(r.status, 400, "{}", r.body);
    let e = &r.json()["error"];
    assert_eq!(e["type"], "invalid_request_error", "{e}");
    assert!(
        e["message"]
            .as_str()
            .is_some_and(|m| m.contains("n_predict")),
        "{e}"
    );
    let whole = post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 2.0, "temperature": 0}),
    );
    assert_eq!(whole.status, 200, "{}", whole.body);
    assert_eq!(whole.json()["tokens_predicted"], 2);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_stop_with_a_non_string_is_refused() {
    let addr = start(4096);
    let r = post(
        addr,
        "/completion",
        &json!({"prompt": "ab", "stop": ["a", 3]}),
    );
    assert_eq!(r.status, 400, "{}", r.body);
    let e = &r.json()["error"];
    assert_eq!(e["type"], "invalid_request_error", "{e}");
    assert!(
        e["message"].as_str().is_some_and(|m| m.contains("stop")),
        "{e}"
    );
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slots_n_remain_during_a_run() {
    let latch = std::sync::Arc::new(Latch::default());
    // Next #1 answers the prompt, #2 feeds the first generated token; #3 waits
    // with two generated.
    let addr = common::start_with(Box::new(Held::new(4096, 3, latch.clone())));
    let gen_thread = std::thread::spawn(move || {
        post(
            addr,
            "/completion",
            &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
        )
    });
    assert!(
        latch.wait_entered(std::time::Duration::from_secs(10)),
        "the generation never reached next #3"
    );
    let s = get(addr, "/slots").json();
    latch.release();
    let t = &s[0]["next_token"];
    assert_eq!(t["n_decoded"], 2, "{s}");
    assert_eq!(t["n_remain"], 3, "{s}");
    assert_eq!(t["has_next_token"], true, "{s}");
    let done = gen_thread.join().expect("generation thread");
    assert_eq!(done.json()["tokens_predicted"], 5);
    let after = get(addr, "/slots").json();
    assert_eq!(after[0]["next_token"]["n_remain"], -1, "{after}");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_metrics_n_decode_total() {
    let addr = start(4096);
    // Five generated: the prompt's decode gives the first, four more give the rest.
    post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}),
    );
    assert_eq!(metric(&get(addr, "/metrics").body, "n_decode_total"), 5.0);
    // None generated: the prompt is still decoded once.
    post(
        addr,
        "/completion",
        &json!({"prompt": "abcabc", "n_predict": 0, "temperature": 0}),
    );
    assert_eq!(metric(&get(addr, "/metrics").body, "n_decode_total"), 6.0);
}

/// Qwen3's chat template renders as jinja2 does: every case of the fixture
/// through `/apply-template` equals its reference render byte for byte. The
/// cases reach `messages[::-1]`, `split` and `strip('\n')` on a think span, and
/// `enable_thinking is false`.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen3_template_renders_as_jinja2() {
    let addr = common::start_templated(
        Box::new(serve::MockEngine::new(4096)),
        include_str!("fixtures/qwen3-chat-template.jinja"),
    );
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/qwen3-renders.json")).expect("fixture JSON");
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 4, "{fixture}");
    for case in cases {
        let r = post(addr, "/apply-template", &case["body"]);
        assert_eq!(r.status, 200, "{}: {}", case["name"], r.body);
        assert_eq!(r.json()["prompt"], case["prompt"], "{}", case["name"]);
    }
}

/// What request A leaves in the cache and request B shares with it (see
/// `hw_cache_prompt_reuses_the_common_prefix`): A holds 10 ids, B's first 8 are
/// among them.
fn slot_a() -> Value {
    json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0})
}

fn slot_b() -> Value {
    json!({"prompt": "abcabcabb", "n_predict": 4, "temperature": 0, "return_tokens": true})
}

fn slot_post(addr: std::net::SocketAddr, action: &str, filename: &str) -> common::Reply {
    post(
        addr,
        &format!("/slots/0?action={action}"),
        &json!({ "filename": filename }),
    )
}

fn erase(addr: std::net::SocketAddr) -> common::Reply {
    call(addr, "POST", "/slots/0?action=erase", None)
}

fn kv_tokens(addr: std::net::SocketAddr) -> f64 {
    metric(&get(addr, "/metrics").body, "kv_cache_tokens")
}

/// The files in `dir`, sorted.
fn listing(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// Save, erase, restore: llama-server's response objects, the file's bytes, and
/// a restored slot that serves the next request from the restored cache, in the
/// same server and in a second one reading the same directory.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slot_save_erase_restore_round_trip() {
    let dir = common::fresh_dir("round-trip");
    let fresh = post(start(4096), "/completion", &slot_b()).json();
    assert_eq!(fresh["timings"]["cache_n"], 0, "{fresh}");
    let addr = common::start_slots(Box::new(serve::MockEngine::new(4096)), &dir);
    post(addr, "/completion", &slot_a());

    let r = slot_post(addr, "save", "a.bin");
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    // Derived: a 24-byte header and 10 ids, then the mock's 16-byte head and 10 ids.
    assert_eq!(v["id_slot"], 0, "{v}");
    assert_eq!(v["filename"], "a.bin", "{v}");
    assert_eq!(v["n_saved"], 10, "{v}");
    assert_eq!(v["n_written"], 120, "{v}");
    assert!(v["timings"]["save_ms"].is_f64(), "{v}");
    let on_disk = std::fs::metadata(dir.join("a.bin")).expect("a.bin").len();
    assert_eq!(on_disk, 120, "the file's bytes against n_written");
    assert_eq!(listing(&dir), ["a.bin"], "a partial file was left behind");

    let r = erase(addr);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json(), json!({"id_slot": 0, "n_erased": 10}));
    assert_eq!(kv_tokens(addr), 0.0);
    let cold = post(addr, "/completion", &slot_b()).json();
    assert_eq!(
        cold["timings"]["cache_n"], 0,
        "erase kept the cache: {cold}"
    );
    assert_eq!(cold["tokens"], fresh["tokens"]);
    // B leaves its 9 prompt ids and 3 of its 4 generated ones.
    assert_eq!(erase(addr).json()["n_erased"], 12);

    let r = slot_post(addr, "restore", "a.bin");
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(v["id_slot"], 0, "{v}");
    assert_eq!(v["filename"], "a.bin", "{v}");
    assert_eq!(v["n_restored"], 10, "{v}");
    assert_eq!(v["n_read"], 120, "{v}");
    assert!(v["timings"]["restore_ms"].is_f64(), "{v}");
    let warm = post(addr, "/completion", &slot_b()).json();
    assert_eq!(
        warm["timings"]["cache_n"], 8,
        "the restored slot must serve B's shared prefix: {warm}"
    );
    assert_eq!(warm["timings"]["prompt_n"], 1, "{warm}");
    assert_eq!(
        warm["tokens"], fresh["tokens"],
        "warm {warm}\nfresh {fresh}"
    );

    let other = common::start_slots(Box::new(serve::MockEngine::new(4096)), &dir);
    let r = slot_post(other, "restore", "a.bin");
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["n_restored"], 10);
    assert_eq!(kv_tokens(other), 10.0);
    let warm = post(other, "/completion", &slot_b()).json();
    assert_eq!(warm["timings"]["cache_n"], 8, "{warm}");
    assert_eq!(
        warm["tokens"], fresh["tokens"],
        "warm {warm}\nfresh {fresh}"
    );
    common::drop_dir(&dir);
}

fn assert_error(r: &common::Reply, status: u16, kind: &str, part: &str) {
    assert_eq!(r.status, status, "{}", r.body);
    let e = &r.json()["error"];
    assert_eq!(e["code"], status, "{e}");
    assert_eq!(e["type"], kind, "{e}");
    assert!(
        e["message"].as_str().is_some_and(|m| m.contains(part)),
        "`{part}` not in {e}"
    );
}

/// Every refusal of a slot action, each by name: the server keeps serving and,
/// where the file was refused before the engine read it, keeps its cache.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slot_action_errors() {
    let bad = |addr, path: &str, body: Option<&str>| call(addr, "POST", path, body);
    // No --slot-save-path: llama-server's 501.
    let plain = start(4096);
    let r = slot_post(plain, "save", "a.bin");
    assert_error(&r, 501, "not_supported_error", "`--slot-save-path`");

    let dir = common::fresh_dir("errors");
    let addr = common::start_slots(Box::new(serve::MockEngine::new(4096)), &dir);
    let e400 = "invalid_request_error";
    assert_error(
        &bad(addr, "/slots/x?action=erase", None),
        400,
        e400,
        "Invalid slot ID",
    );
    assert_error(
        &slot_post_at(addr, 1, "save", "a.bin"),
        400,
        e400,
        "Invalid slot ID",
    );
    assert_error(
        &bad(addr, "/slots/0?action=frob", None),
        400,
        e400,
        "Invalid action",
    );
    assert_error(&bad(addr, "/slots/0", None), 400, e400, "Invalid action");
    // The query string is percent-decoded: `er%61se` is `erase`, and a broken
    // escape is refused by name.
    let r = bad(addr, "/slots/0?action=er%61se", None);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json(), json!({"id_slot": 0, "n_erased": 0}));
    assert_error(
        &bad(addr, "/slots/0?action=%zz", None),
        400,
        e400,
        "percent-encoding",
    );
    assert_error(
        &bad(addr, "/slots/0?action=save", Some("{}")),
        400,
        e400,
        "filename",
    );
    assert_error(
        &bad(addr, "/slots/0?action=save", Some(r#"{"filename": 3}"#)),
        400,
        e400,
        "filename",
    );
    for name in ["sub/a.bin", "../a.bin", "a\\b", ".."] {
        assert_error(
            &slot_post(addr, "save", name),
            400,
            e400,
            "Invalid filename",
        );
    }
    assert_error(
        &slot_post(addr, "restore", "none.bin"),
        400,
        e400,
        "Unable to restore slot",
    );
    assert!(listing(&dir).is_empty(), "{:?}", listing(&dir));

    // A file this server did not write: refused before the engine reads, so
    // the cache A left serves B.
    post(addr, "/completion", &slot_a());
    assert_eq!(slot_post(addr, "save", "v.bin").status, 200);
    let good = std::fs::read(dir.join("v.bin")).expect("v.bin");
    let patched = |name: &str, at: usize, bytes: &[u8]| {
        let mut f = good.clone();
        f[at..at + bytes.len()].copy_from_slice(bytes);
        std::fs::write(dir.join(name), f).expect("write");
    };
    patched("v99.bin", 8, &99u32.to_le_bytes());
    assert_error(
        &slot_post(addr, "restore", "v99.bin"),
        400,
        e400,
        "slot file version 99",
    );
    patched("magic.bin", 0, b"X");
    assert_error(
        &slot_post(addr, "restore", "magic.bin"),
        400,
        e400,
        "not a slot file",
    );
    std::fs::write(dir.join("short.bin"), &good[..30]).expect("write");
    assert_error(
        &slot_post(addr, "restore", "short.bin"),
        400,
        e400,
        "Unable to restore slot",
    );
    assert_eq!(
        kv_tokens(addr),
        10.0,
        "a refused header must leave the cache"
    );
    let kept = post(addr, "/completion", &slot_b()).json();
    assert_eq!(kept["timings"]["cache_n"], 8, "{kept}");

    // The engine's part refused: the engine read, so the cache is reset.
    post(addr, "/completion", &slot_a());
    patched("mocktag.bin", 64, b"X");
    assert_error(
        &slot_post(addr, "restore", "mocktag.bin"),
        400,
        e400,
        "mock state",
    );
    assert_eq!(
        kv_tokens(addr),
        0.0,
        "a refused engine state must empty the slot"
    );
    let reset = post(addr, "/completion", &slot_b()).json();
    assert_eq!(reset["timings"]["cache_n"], 0, "{reset}");
    let mut long = good.clone();
    long.push(0);
    std::fs::write(dir.join("long.bin"), long).expect("write");
    assert_error(
        &slot_post(addr, "restore", "long.bin"),
        400,
        e400,
        "runs on past",
    );

    // An engine with the trait's defaults refuses by name and keeps serving.
    let unsupported = common::start_slots(Box::new(NoCut(serve::MockEngine::new(4096))), &dir);
    post(unsupported, "/completion", &slot_a());
    let before = listing(&dir);
    let r = slot_post(unsupported, "save", "u.bin");
    assert_error(
        &r,
        501,
        "not_supported_error",
        "does not support slot save/restore",
    );
    assert_eq!(listing(&dir), before, "a refused save left a file");
    let r = slot_post(unsupported, "restore", "v.bin");
    assert_error(
        &r,
        501,
        "not_supported_error",
        "does not support slot save/restore",
    );
    assert_eq!(get(unsupported, "/health").status, 200);
    assert_eq!(erase(unsupported).json()["n_erased"], 10);

    // The flag names a directory that must exist.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_bloomery-serve"))
        .args(["--port", "0", "--slot-save-path"])
        .arg(dir.join("missing"))
        .output()
        .expect("run bloomery-serve");
    assert_eq!(out.status.code(), Some(64), "{out:?}");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("not a directory"), "{err}");
    common::drop_dir(&dir);
}

fn slot_post_at(
    addr: std::net::SocketAddr,
    id: i64,
    action: &str,
    filename: &str,
) -> common::Reply {
    post(
        addr,
        &format!("/slots/{id}?action={action}"),
        &json!({ "filename": filename }),
    )
}

/// A slot action while a request runs on the slot is a 503; once it is done the
/// same action goes through.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slot_action_on_a_busy_slot_is_503() {
    let dir = common::fresh_dir("busy");
    let latch = std::sync::Arc::new(Latch::default());
    let addr = common::start_slots(Box::new(Held::new(4096, 3, latch.clone())), &dir);
    let gen_thread = std::thread::spawn(move || post(addr, "/completion", &slot_a()));
    assert!(
        latch.wait_entered(std::time::Duration::from_secs(10)),
        "the generation never reached next #3"
    );
    let save = slot_post(addr, "save", "busy.bin");
    let wipe = erase(addr);
    latch.release();
    assert_error(&save, 503, "unavailable_error", "processing a request");
    assert_error(&wipe, 503, "unavailable_error", "processing a request");
    assert_eq!(gen_thread.join().expect("generation").status, 200);
    assert!(listing(&dir).is_empty(), "{:?}", listing(&dir));
    let r = slot_post(addr, "save", "busy.bin");
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["n_saved"], 10);
    common::drop_dir(&dir);
}
