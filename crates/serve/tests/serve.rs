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
        (
            "tools",
            json!([{"type": "function", "function": {"name": "f"}}]),
        ),
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
        ("tool_choice", json!("none")),
        ("cache_prompt", json!(true)),
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
