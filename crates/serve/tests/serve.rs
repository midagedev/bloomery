//! Gate: the HTTP server over the mock engine — JSON shapes, stream framing and
//! stop rules against llama-server's field names (`tests/fixtures`).
//!
//! The mock's greedy output is derived by hand from its bigram rule (see
//! `serve::mock`): for the chat below the rendered prompt ends in `</think>`, whose
//! earlier occurrence is followed by `abc<｜end▁of▁sentence｜>`, so greedy decoding
//! yields `a`, `b`, `c`, EOS.

mod common;
mod slots;

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

/// A chat content part this server cannot read is refused, never dropped: a
/// media part by its kind, any other by its type, on both paths that render a
/// chat, in llama-server's words; a `/completion` prompt object carrying media
/// likewise. Text parts still render byte for byte as the string they join to.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_media_parts_are_refused() {
    let addr = start(4096);
    let refused = |r: common::Reply, message: &str| {
        assert_eq!(r.status, 400, "{}", r.body);
        let e = &r.json()["error"];
        assert_eq!(e["code"], 400, "{e}");
        assert_eq!(e["type"], "invalid_request_error", "{e}");
        assert_eq!(e["message"], message, "{e}");
    };
    let user = |content: Value| json!({"messages": [{"role": "user", "content": content}]});
    let hint =
        "input is not supported - hint: if this is unexpected, you may need to provide the mmproj";
    let parts = [
        (
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}),
            format!("image {hint}"),
        ),
        (
            json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}),
            format!("audio {hint}"),
        ),
        (
            json!({"type": "bogus"}),
            "unsupported content[].type".to_owned(),
        ),
    ];
    for (part, message) in parts {
        let b = chat_body(user(json!([{"type": "text", "text": "hi"}, part])));
        for path in ["/v1/chat/completions", "/apply-template"] {
            refused(post(addr, path, &b), &message);
        }
    }
    refused(
        post(
            addr,
            "/completion",
            &json!({"prompt": {"prompt_string": "ab", "multimodal_data": ["AAAA"]}}),
        ),
        "Multimodal data provided, but model does not support multimodal requests.",
    );
    let rendered = |content: Value| {
        let r = post(addr, "/apply-template", &user(content));
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["prompt"].clone()
    };
    assert_eq!(
        rendered(json!([{"type": "text", "text": "hi"}])),
        rendered(json!("hi"))
    );
    let ab = json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]);
    assert_eq!(rendered(ab.clone()), rendered(json!("a\nb")));
    let reply = |content: Value| {
        let mut b = chat_body(user(content));
        b["max_tokens"] = json!(4);
        let r = post(addr, "/v1/chat/completions", &b);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let u = &v["usage"];
        let tokens = (u["prompt_tokens"].clone(), u["completion_tokens"].clone());
        (v["choices"][0]["message"].clone(), tokens)
    };
    assert_eq!(reply(ab), reply(json!("a\nb")));
}

/// A template that renders each message's content and nothing else: the
/// prompt is the user's text, so the mock's reply reads the image's span.
const CONTENT_TEMPLATE: &str = "{%- for m in messages -%}{{- m['content'] -}}{%- endfor -%}";

/// 2×1 PNGs of the colours (200, 100, 50) and (50, 100, 200): two images of
/// one size, so one span length (the mock's model: a position a pixel).
const PNG_A: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg==";
const PNG_B: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGMwSjkBRAAIbQK9i80DoQAAAABJRU5ErkJggg==";

fn png_url(b64: &str) -> String {
    format!("data:image/png;base64,{b64}")
}

/// The media mock's server on `template`, and the log of its image calls.
fn media_server(
    template: &str,
) -> (
    std::net::SocketAddr,
    std::sync::Arc<std::sync::Mutex<Vec<serve::mock::MediaCall>>>,
) {
    let (mock, log) = serve::MockEngine::new(4096).with_media();
    (common::start_templated(Box::new(mock), template), log)
}

fn media_calls(log: &std::sync::Mutex<Vec<serve::mock::MediaCall>>) -> Vec<serve::mock::MediaCall> {
    std::mem::take(&mut *log.lock().expect("the media log"))
}

/// A greedy chat of four tokens whose one user message is `parts`.
fn parts_chat(parts: Value) -> Value {
    json!({
        "messages": [{"role": "user", "content": parts}],
        "temperature": 0,
        "max_tokens": 4,
    })
}

/// `q`, the image at `url`, and an empty text part: the prompt `q\n` image
/// `\n` under [`CONTENT_TEMPLATE`], whose reply starts with the span.
fn image_chat(url: &str) -> Value {
    parts_chat(json!([
        {"type": "text", "text": "q"},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": ""},
    ]))
}

/// One chat's reply text and its cached prompt tokens.
fn chat_reply(addr: std::net::SocketAddr, body: &Value) -> (Value, Value) {
    let r = post(addr, "/v1/chat/completions", body);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    (
        v["choices"][0]["message"]["content"].clone(),
        v["usage"]["prompt_tokens_details"]["cached_tokens"].clone(),
    )
}

/// A chat's image part becomes the model's placeholder in the render, one
/// token the request expands to the image's span (the mock's: two positions
/// for 2×1); `/props` names the vision, `/apply-template` returns the
/// placeholder unexpanded, and the engine is fed the image with the ids that
/// hold its span.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_images_expand_to_spans() {
    use serve::Tokenizer;
    use serve::mock::{IMAGE_ID, IMAGE_SPECIAL, MediaCall, MediaTokenizer};
    let (addr, log) = media_server(common::V41_TEMPLATE);
    assert_eq!(get(addr, "/props").json()["modalities"]["vision"], true);
    let url = png_url(PNG_A);
    let body = parts_chat(json!([
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": url},
        {"type": "text", "text": "please"},
    ]));
    let r = post(addr, "/apply-template", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let text = r.json()["prompt"].as_str().expect("a prompt").to_owned();
    assert!(
        text.contains(&format!("look\n{IMAGE_SPECIAL}\nplease")),
        "the parts joined by the model's separator, the image unexpanded: {text}"
    );
    let rendered = MediaTokenizer.encode(&text);
    let at = rendered
        .iter()
        .position(|&t| t == IMAGE_ID)
        .expect("the placeholder");
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let n = r.json()["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");
    assert_eq!(
        n as usize,
        rendered.len() + 1,
        "one placeholder, two positions"
    );
    let mut ids = rendered.clone();
    ids.insert(at, IMAGE_ID);
    ids.pop();
    let key = serve::media::load(&url).expect("the fixture").key;
    assert_eq!(
        media_calls(&log),
        [MediaCall {
            ids,
            spans: vec![serve::media::MediaSpan { at, len: 2, key }],
        }],
        "the prompt but its last id in one call, the span at the placeholder"
    );
}

/// An Anthropic `image` block feeds the engine as the chat's `image_url` part
/// does: on two fresh servers, `/v1/messages` and `/v1/chat/completions` of
/// the same turn answer 200 and make the same engine call, the image's span
/// at its placeholder.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_images_feed_as_the_chat_does() {
    let anthropic = json!({
        "max_tokens": 4,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_A}},
        ]}],
    });
    let chat = parts_chat(json!([
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": png_url(PNG_A)}},
    ]));
    let (a, a_log) = media_server(common::V41_TEMPLATE);
    let r = post(a, "/v1/messages", &anthropic);
    assert_eq!(r.status, 200, "{}", r.body);
    let (o, o_log) = media_server(common::V41_TEMPLATE);
    let r = post(o, "/v1/chat/completions", &chat);
    assert_eq!(r.status, 200, "{}", r.body);
    let calls = media_calls(&a_log);
    assert_eq!(calls.len(), 1, "one engine call with the image: {calls:?}");
    assert_eq!(calls, media_calls(&o_log), "the chat path's call");
}

/// The prompt cache keys an image's span by the image: two chats of one text
/// whose images differ only in their pixels share their ids (every span
/// position carries the image token), and the second keeps up to the span's
/// start, feeds its own image and replies as a fresh server does; the same
/// image again keeps the span whole and feeds none. The mock's reply spells
/// the span it holds, so a kept span of the other image would show.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_reuse_needs_the_same_image() {
    use serve::mock::{IMAGE_ID, MediaCall};
    let (addr, log) = media_server(CONTENT_TEMPLATE);
    let (url_a, url_b) = (png_url(PNG_A), png_url(PNG_B));
    let first = image_chat(&url_a);
    let (reply_a, cached) = chat_reply(addr, &first);
    assert_eq!(cached, 0);
    assert_eq!(media_calls(&log).len(), 1);
    // The same image, then the reply and a new user turn: `q\n`, the span
    // (2..4), `\n`, the reply's ids but its last, then the new text.
    let mut longer = first.clone();
    let msgs = longer["messages"].as_array_mut().expect("messages");
    msgs.push(json!({"role": "assistant", "content": reply_a}));
    msgs.push(json!({"role": "user", "content": "more\n"}));
    let (_, cached) = chat_reply(addr, &longer);
    assert_eq!(
        cached, 8,
        "the first prompt and three reply ids: past the span"
    );
    assert_eq!(media_calls(&log), [], "a span kept whole is not fed again");
    let other = image_chat(&url_b);
    let (reply_b, cached) = chat_reply(addr, &other);
    assert_eq!(cached, 2, "another image of one size: the span's start");
    let key_b = serve::media::load(&url_b).expect("the fixture").key;
    assert_eq!(
        media_calls(&log),
        [MediaCall {
            ids: vec![IMAGE_ID; 2],
            spans: vec![serve::media::MediaSpan {
                at: 0,
                len: 2,
                key: key_b,
            }],
        }],
        "the other image fed from its span's start"
    );
    let (fresh, _) = media_server(CONTENT_TEMPLATE);
    assert_eq!(chat_reply(fresh, &other).0, reply_b, "b's reply alone");
    assert_eq!(chat_reply(fresh, &first).0, reply_a, "a's reply alone");
    assert_ne!(
        reply_a, reply_b,
        "the replies spell their images: {reply_a}"
    );
}

/// Every refusal of a chat's image, each a 400 by name: a URL this server
/// does not fetch, a base64 text that is not one, bytes of another format
/// than the URL declares, an image outside a user message, a placeholder
/// typed into a text part or into a string content (it pairs with no image),
/// and a prompt that ends with an image; `/completion` refuses the typed
/// placeholder too. The engine, which takes an image token only with its
/// image, serves on after all of them.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_image_refusals_are_named() {
    use serve::mock::IMAGE_SPECIAL;
    let (addr, log) = media_server(CONTENT_TEMPLATE);
    let refused = |path: &str, body: &Value, part: &str| {
        let r = post(addr, path, body);
        assert_eq!(r.status, 400, "{}", r.body);
        let e = &r.json()["error"];
        assert_eq!(e["type"], "invalid_request_error", "{e}");
        assert!(
            e["message"].as_str().is_some_and(|m| m.contains(part)),
            "`{part}` not in {e}"
        );
    };
    let chat = "/v1/chat/completions";
    let png = png_url(PNG_A);
    refused(
        chat,
        &image_chat("https://example.com/a.png"),
        "URLs are not fetched",
    );
    refused(
        chat,
        &image_chat("data:image/png;base64,Zm9vYmF"),
        "base64:",
    );
    refused(
        chat,
        &image_chat(&png.replace("image/png", "image/jpeg")),
        "declared JPEG, but the bytes are PNG",
    );
    let assistant = json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "image_url", "image_url": png}]},
        {"role": "user", "content": "hi"},
    ]});
    refused(chat, &assistant, "only user messages carry images");
    let typed = format!("an {IMAGE_SPECIAL} by hand\n");
    refused(
        chat,
        &parts_chat(json!([{"type": "text", "text": typed}])),
        "carries the image placeholder",
    );
    refused(
        chat,
        &parts_chat(json!(typed)),
        "1 image placeholder(s) in the prompt for 0 image(s)",
    );
    let with_system = json!({"messages": [
        {"role": "system", "content": typed},
        {"role": "user", "content": image_chat(&png)["messages"][0]["content"]},
    ]});
    refused(
        chat,
        &with_system,
        "2 image placeholder(s) in the prompt for 1 image(s)",
    );
    let last = parts_chat(json!([
        {"type": "text", "text": "q"},
        {"type": "image_url", "image_url": png},
    ]));
    refused(chat, &last, "the prompt ends with an image");
    refused(
        "/completion",
        &json!({"prompt": typed, "n_predict": 2}),
        "1 image placeholder(s) in the prompt for 0 image(s)",
    );
    assert_eq!(media_calls(&log), [], "nothing reached the engine");
    let (reply, _) = chat_reply(addr, &image_chat(&png));
    assert!(reply.as_str().is_some_and(|r| !r.is_empty()), "{reply}");
    assert_eq!(get(addr, "/health").status, 200);
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

/// The prompt entries the engine saw, in order, and the flag that ends the
/// hold on the slot: until it is set every entry blocks where it is.
#[derive(Default)]
struct EntryLog {
    state: std::sync::Mutex<(Vec<String>, bool)>,
    cv: std::sync::Condvar,
}

impl EntryLog {
    fn enter(&self, prompt: String) {
        let mut g = self.state.lock().expect("entry log");
        g.0.push(prompt);
        self.cv.notify_all();
        while !g.1 {
            g = self.cv.wait(g).expect("entry log");
        }
    }

    fn len_reached(&self, n: usize, bound: std::time::Duration) -> bool {
        let g = self.state.lock().expect("entry log");
        let (g, _) = self
            .cv
            .wait_timeout_while(g, bound, |s| s.0.len() < n)
            .expect("entry log");
        g.0.len() >= n
    }

    fn release(&self) {
        self.state.lock().expect("entry log").1 = true;
        self.cv.notify_all();
    }

    fn prompts(&self) -> Vec<String> {
        self.state.lock().expect("entry log").0.clone()
    }
}

/// The mock engine with [`EntryLog`] in `prefill`: every prompt's entry is
/// recorded in order and, until the log is released, blocks there, so one
/// request holds the slot while later ones queue behind it.
struct Queued {
    inner: serve::MockEngine,
    log: std::sync::Arc<EntryLog>,
}

impl Queued {
    fn new(ctx: usize, log: std::sync::Arc<EntryLog>) -> Queued {
        Queued {
            inner: serve::MockEngine::new(ctx),
            log,
        }
    }
}

impl serve::Engine for Queued {
    fn tokenizer(&self) -> std::sync::Arc<dyn serve::Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), serve::EngineError> {
        self.log.enter(self.inner.tokenizer().decode(ids));
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, serve::EngineError> {
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
}

/// Waits until `requests_deferred` reaches `n`: `n` requests have drawn a
/// ticket and not got the slot.
fn wait_deferred(addr: std::net::SocketAddr, n: usize, bound: std::time::Duration) {
    let deadline = std::time::Instant::now() + bound;
    loop {
        let deferred = metric(&get(addr, "/metrics").body, "requests_deferred");
        if deferred as usize >= n {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "requests_deferred never reached {n} (now {deferred}): a request never queued"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Requests queued behind a held slot get it in the order they arrived: the
/// order their tickets were drawn.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_queued_requests_get_the_slot_in_arrival_order() {
    const QUEUED: usize = 4;
    let log = std::sync::Arc::new(EntryLog::default());
    let addr = common::start_with(Box::new(Queued::new(4096, log.clone())));
    let (tx, rx) = std::sync::mpsc::channel();
    let mut workers = Vec::new();
    for i in 0..=QUEUED {
        let (tx, addr) = (tx.clone(), addr);
        workers.push(std::thread::spawn(move || {
            // Two tokens: `prefill` sees the prompt less its last token, so a
            // one-token prompt would arrive there empty.
            let body = json!({"prompt": format!("{i}a"), "n_predict": 1, "temperature": 0});
            let r = post(addr, "/completion", &body);
            let _ = tx.send((i, r.status));
        }));
        if i == 0 {
            assert!(
                log.len_reached(1, std::time::Duration::from_secs(10)),
                "the first request never reached the engine; entries so far: {:?}",
                log.prompts()
            );
        } else {
            // Confirmed queued before the next starts, so the arrival order
            // is the spawn order.
            wait_deferred(addr, i, std::time::Duration::from_secs(10));
        }
    }
    log.release();
    for _ in 0..=QUEUED {
        let (i, status) = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap_or_else(|e| {
                panic!(
                    "a request never finished ({e}); entries so far: {:?}",
                    log.prompts()
                )
            });
        assert_eq!(status, 200, "request {i}");
    }
    for w in workers {
        w.join().expect("worker");
    }
    let want: Vec<String> = (0..=QUEUED).map(|i| i.to_string()).collect();
    assert_eq!(log.prompts(), want, "the slot must pass in arrival order");
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
    for key in ["model", "placement", "draft", "ctx_verified"] {
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

/// GLM-5.3-Flash's chat template renders as jinja2 does: every case of the
/// fixture through `/apply-template` equals its reference render byte for
/// byte, and the case jinja2 refuses (tool-call `arguments` given as a JSON
/// string, which the template walks with `.items()`) is refused by name. The
/// cases reach the template's twelve macros (called in output, in `+`, in
/// `==` and as conditions), `break` in three loops, `| capitalize` on the
/// reasoning effort, `tojson(ensure_ascii=False)`, `clear_thinking`, and the
/// tool-result reordering by call id.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm5_template_renders_as_jinja2() {
    let addr = common::start_templated(
        Box::new(serve::MockEngine::new(4096)),
        include_str!("fixtures/glm5-chat-template.jinja"),
    );
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/glm5-renders.json")).expect("fixture JSON");
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 12, "{fixture}");
    for case in cases {
        let r = post(addr, "/apply-template", &case["body"]);
        if case["error"].is_string() {
            assert_eq!(r.status, 500, "{}: {}", case["name"], r.body);
            let message = r.json()["error"]["message"].clone();
            assert!(
                message.as_str().is_some_and(|m| m.contains("items()")),
                "{}: {message} does not name items()",
                case["name"]
            );
            continue;
        }
        assert_eq!(r.status, 200, "{}: {}", case["name"], r.body);
        assert_eq!(r.json()["prompt"], case["prompt"], "{}", case["name"]);
    }
}

/// Qwen3.8's chat template renders as jinja2 does: every case of the fixture
/// through `/apply-template` equals its reference render byte for byte, and
/// the case the template refuses (tool-call `arguments` given as a JSON
/// string) is refused by name. The cases reach `reasoning_effort|default`
/// against a tuple, the reasoning effort's system turn, and the tool calls.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen38_template_renders_as_jinja2() {
    let addr = common::start_templated(
        Box::new(serve::MockEngine::new(4096)),
        include_str!("fixtures/qwen38-chat-template.jinja"),
    );
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/qwen38-renders.json")).expect("fixture JSON");
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 12, "{fixture}");
    for case in cases {
        let r = post(addr, "/apply-template", &case["body"]);
        if case["error"].is_string() {
            assert_eq!(r.status, 500, "{}: {}", case["name"], r.body);
            let message = r.json()["error"]["message"].clone();
            assert!(
                message
                    .as_str()
                    .is_some_and(|m| m.contains("passed as a JSON string")),
                "{}: {message} does not name the JSON string",
                case["name"]
            );
            continue;
        }
        assert_eq!(r.status, 200, "{}: {}", case["name"], r.body);
        assert_eq!(r.json()["prompt"], case["prompt"], "{}", case["name"]);
    }
}

/// Every id the vocabulary names as a stop ends a generation, not only its
/// EOS: with GLM-5.3-Flash's three (the header's eos, eot and eom), a
/// generation that emits any one of them stops there, reports `stopped_eos`,
/// and counts the stop among its tokens. The script runs on past each stop,
/// so a loop that stopped on the EOS alone would run to `n_predict` on the
/// other two.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_every_stop_id_ends_a_generation() {
    use serve::{MockTokenizer, ScriptedEngine, Tokenizer};
    const STOPS: [u32; 3] = [154_820, 154_827, 154_829];
    let (a, b) = (MockTokenizer.encode("a")[0], MockTokenizer.encode("b")[0]);
    for stop in STOPS {
        let script = vec![a, stop, b, b, b, b, b, b];
        let engine = ScriptedEngine::from_ids(4096, script).with_stops(&STOPS);
        let addr = common::start_with(Box::new(engine));
        let r = post(
            addr,
            "/completion",
            &json!({"prompt": "hi", "n_predict": 6, "temperature": 0}),
        )
        .json();
        assert_eq!(r["stopped_eos"], true, "stop {stop}: {r}");
        assert_eq!(r["stopped_limit"], false, "stop {stop}: {r}");
        assert_eq!(r["tokens_predicted"], 2, "stop {stop}: {r}");
        assert_eq!(r["content"], "a", "stop {stop}: {r}");
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
    // Derived: a 24-byte header, 10 ids and an empty span table (its count),
    // then the mock's 16-byte head and 10 ids.
    let bytes = 24 + 4 * 10 + 8 + 16 + 4 * 10;
    assert_eq!(v["id_slot"], 0, "{v}");
    assert_eq!(v["filename"], "a.bin", "{v}");
    assert_eq!(v["n_saved"], 10, "{v}");
    assert_eq!(v["n_written"], bytes, "{v}");
    assert!(v["timings"]["save_ms"].is_f64(), "{v}");
    let on_disk = std::fs::metadata(dir.join("a.bin")).expect("a.bin").len();
    assert_eq!(on_disk, bytes, "the file's bytes against n_written");
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
    assert_eq!(v["n_read"], bytes, "{v}");
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

/// A slot file carries the slot's images: the span table after the ids, and
/// a restore — in the server that saved it and in another — keeps the span
/// whole for the next chat of the same image and feeds none. The same file
/// as version 1 (ids alone) restores a slot of no image, so the same chat
/// keeps up to the span's start and feeds the image again.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_slot_files_carry_image_spans() {
    let dir = common::fresh_dir("media-slots");
    let start_media = || {
        let (mock, log) = serve::MockEngine::new(4096).with_media();
        (common::start_slots(Box::new(mock), &dir), log)
    };
    let (addr, log) = start_media();
    let body = image_chat(&png_url(PNG_A));
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let n = r.json()["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");
    let at = media_calls(&log)[0].spans[0].at as u64;
    let v = slot_post(addr, "save", "media.bin").json();
    let held = v["n_saved"].as_u64().expect("n_saved");
    // Derived: a 24-byte header, the held ids, the span count and one span,
    // then the mock's 16-byte head and its context of as many ids.
    let bytes = 24 + 4 * held + 8 + 48 + 16 + 4 * held;
    assert_eq!(v["n_written"], bytes, "{v}");
    let warm = |addr| {
        let r = post(addr, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["usage"]["prompt_tokens_details"]["cached_tokens"].clone()
    };

    assert_eq!(erase(addr).status, 200);
    let r = slot_post(addr, "restore", "media.bin");
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["n_restored"], held);
    assert_eq!(warm(addr), n - 1, "the restored span is kept whole");
    assert_eq!(media_calls(&log), [], "a restored span is not fed again");

    let (other, other_log) = start_media();
    assert_eq!(slot_post(other, "restore", "media.bin").status, 200);
    assert_eq!(warm(other), n - 1, "another server keeps the span whole");
    assert_eq!(media_calls(&other_log), []);

    let good = std::fs::read(dir.join("media.bin")).expect("media.bin");
    let ids_end = (24 + 4 * held) as usize;
    let mut v1 = good[..ids_end].to_vec();
    v1[8..12].copy_from_slice(&1u32.to_le_bytes());
    v1.extend_from_slice(&good[ids_end + 8 + 48..]);
    std::fs::write(dir.join("v1.bin"), &v1).expect("write");
    assert_eq!(erase(other).status, 200);
    let r = slot_post(other, "restore", "v1.bin");
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["n_restored"], held);
    assert_eq!(warm(other), at, "a slot of no image: the span's start");
    assert_eq!(media_calls(&other_log).len(), 1, "the image fed again");
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

    // The engine's part refused: the engine read, so the cache is reset. Its
    // tag follows the 24-byte header, A's 10 ids and the span count.
    post(addr, "/completion", &slot_a());
    patched("mocktag.bin", 24 + 4 * 10 + 8, b"X");
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

/// A tool schema reaches the template with its keys in the request's order:
/// Qwen3's template prints each tool with `tojson`, and jinja2 3.1.6 (the HF
/// environment) keeps the order the client sent, `properties` `z` before `a`
/// included. The body is sent as written, not re-serialized.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tool_schema_keys_keep_the_request_order() {
    let addr = common::start_templated(
        Box::new(serve::MockEngine::new(4096)),
        include_str!("fixtures/qwen3-chat-template.jinja"),
    );
    let body = r#"{"messages":[{"role":"user","content":"q"}],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"integer"}},"required":["z"]}}}]}"#;
    let r = call(addr, "POST", "/apply-template", Some(body));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(
        r.json()["prompt"],
        "<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"d\", \"parameters\": {\"type\": \"object\", \"properties\": {\"z\": {\"type\": \"string\"}, \"a\": {\"type\": \"integer\"}}, \"required\": [\"z\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
    );
}

/// A template whose tool-call markup the server does not parse (`tests/common`'s
/// synthetic `NO_MARKUP_TEMPLATE`, the one template left in that state now
/// that every real family — DSML, GLM's, Hermes', Qwen's XML — has its parser
/// and its gate) refuses a chat request that asks for tool calls, by name,
/// instead of returning the calls as `content`; with `tool_choice` `"none"` or
/// no tools the request runs, and `/apply-template` still renders the tools.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tools_under_an_unparsed_markup_are_refused() {
    let addr = common::start_templated(
        Box::new(serve::MockEngine::new(4096)),
        common::NO_MARKUP_TEMPLATE,
    );
    let tools =
        json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
    let r = post(
        addr,
        "/v1/chat/completions",
        &chat_body(json!({ "tools": tools })),
    );
    assert_error(&r, 501, "not_supported_error", "tool-call markup");
    let r = post(
        addr,
        "/v1/chat/completions",
        &chat_body(json!({ "tools": tools, "stream": true })),
    );
    assert_error(&r, 501, "not_supported_error", "tool-call markup");
    for extra in [json!({ "tools": tools, "tool_choice": "none" }), json!({})] {
        let r = post(addr, "/v1/chat/completions", &chat_body(extra.clone()));
        assert_eq!(r.status, 200, "{extra}: {}", r.body);
    }
    let r = post(
        addr,
        "/apply-template",
        &chat_body(json!({ "tools": tools })),
    );
    assert_eq!(r.status, 200, "{}", r.body);
}

/// `ignore_eos` bans every end-of-generation id, as llama-server's
/// `logit_bias_eog`: a script that emits one of GLM's three stops gets the
/// best other id instead (the scripted logits leave every other id tied at
/// the floor, so greedy takes id 0), greedy and sampled alike, and the
/// generation runs to `n_predict`.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ignore_eos_bans_every_stop_id() {
    use serve::{MockTokenizer, ScriptedEngine, Tokenizer};
    const STOPS: [u32; 3] = [154_820, 154_827, 154_829];
    let (a, b) = (MockTokenizer.encode("a")[0], MockTokenizer.encode("b")[0]);
    for stop in STOPS {
        for temperature in [0.0, 1.0] {
            let script = vec![a, stop, b, b, b, b];
            let engine = ScriptedEngine::from_ids(4096, script).with_stops(&STOPS);
            let addr = common::start_with(Box::new(engine));
            let r = post(
                addr,
                "/completion",
                &json!({"prompt": "hi", "n_predict": 4, "temperature": temperature,
                        "seed": 7, "ignore_eos": true, "return_tokens": true}),
            )
            .json();
            let tokens: Vec<u64> = r["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|t| t.as_u64().expect("id"))
                .collect();
            let at = format!("stop {stop}, temperature {temperature}: {r}");
            assert_eq!(tokens.len(), 4, "{at}");
            assert!(
                tokens
                    .iter()
                    .all(|t| !STOPS.iter().any(|&s| u64::from(s) == *t)),
                "{at}"
            );
            assert_eq!(r["stopped_limit"], true, "{at}");
            if temperature == 0.0 {
                assert_eq!(
                    tokens,
                    [u64::from(a), 0, u64::from(b), u64::from(b)],
                    "{at}"
                );
            }
        }
    }
}

/// The mock whose `reset` and `cut` — the prompt cache's engine calls, one of
/// which every request below makes — each take at least [`SLOW`], so a
/// request's `cache_ms` is never 0 by the clock's grain.
struct SlowCut(serve::MockEngine);

const SLOW: std::time::Duration = std::time::Duration::from_millis(5);

impl serve::Engine for SlowCut {
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
        std::thread::sleep(SLOW);
        self.0.reset()
    }
    fn keepable(&self, n: usize) -> usize {
        self.0.keepable(n)
    }
    fn cut(&mut self, n: usize) -> Result<(), serve::EngineError> {
        std::thread::sleep(SLOW);
        self.0.cut(n)
    }
    fn ctx_max(&self) -> usize {
        self.0.ctx_max()
    }
    fn describe(&self) -> String {
        self.0.describe()
    }
}

/// `/metrics`' totals are the sums of the requests' `timings`: the prompt
/// tokens evaluated and kept, the prompt cache's time (ours), and the longest
/// sequence.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_metrics_count_cached_prompt_tokens() {
    let addr = common::start_with(Box::new(SlowCut(serve::MockEngine::new(4096))));
    let run = |body: Value| post(addr, "/completion", &body).json();
    // A leaves "abcabc" and four generated ids; B keeps 8 of its 9; C keeps none
    // and is the shortest, so the largest sequence is not the last one.
    let a = run(json!({"prompt": "abcabc", "n_predict": 5, "temperature": 0}));
    let b = run(json!({"prompt": "abcabcabb", "n_predict": 4, "temperature": 0}));
    let c = run(json!({"prompt": "xyz", "n_predict": 1, "temperature": 0, "cache_prompt": false}));
    let field = |v: &Value, k: &str| v["timings"][k].as_f64().expect(k);
    let replies = [&a, &b, &c];
    assert_eq!(field(&b, "cache_n"), 8.0, "{b}");
    let body = get(addr, "/metrics").body;
    let sum = |k: &str| replies.iter().map(|v| field(v, k)).sum::<f64>();
    assert_eq!(metric(&body, "prompt_tokens_cached_total"), sum("cache_n"));
    assert_eq!(metric(&body, "prompt_tokens_total"), sum("prompt_n"));
    let slow = SLOW.as_secs_f64() * 1e3;
    assert!(
        sum("cache_ms") >= 3.0 * slow,
        "the fixture: each request's cache work takes {slow} ms or more: {}",
        sum("cache_ms")
    );
    assert_eq!(
        metric(&body, "prompt_cache_seconds_total"),
        sum("cache_ms") / 1e3
    );
    let longest = replies
        .iter()
        .map(|v| field(v, "n_past"))
        .fold(0.0, f64::max);
    assert!(field(&c, "n_past") < longest, "{c}");
    assert_eq!(metric(&body, "n_tokens_max"), longest);
    assert!(
        body.contains("# TYPE llamacpp:prompt_tokens_cached_total counter"),
        "{body}"
    );
}

/// What a [`Spy`] engine saw: every `keepable` it answered, `(n, answer)`,
/// oldest first, and the positions it holds.
#[derive(Default)]
struct Log {
    asked: Vec<(usize, usize)>,
    held: usize,
}

type Asked = std::sync::Arc<std::sync::Mutex<Log>>;

/// The mock engine under a keep rule, logging every `keepable` it answers:
/// the reuse tests ask the engine what it keeps instead of modelling its rule.
/// With `tail`, a prompt call (one `prefill`) keeps only its start and its last
/// `tail` positions, as V4.1's prompt-call hole does; without, any prefix.
struct Spy {
    inner: serve::MockEngine,
    tail: Option<usize>,
    /// Each prompt call's start and the end of its hole: a cut strictly
    /// between them falls to the start.
    holes: Vec<(usize, usize)>,
    asked: Asked,
}

impl Spy {
    fn held(&self) -> usize {
        self.asked.lock().expect("log").held
    }

    fn hold(&self, n: usize) {
        self.asked.lock().expect("log").held = n;
    }

    fn rule(&self, n: usize) -> usize {
        let n = n.min(self.held());
        self.holes
            .iter()
            .find(|&&(start, end)| start < n && n < end)
            .map_or(n, |&(start, _)| start)
    }
}

impl serve::Engine for Spy {
    fn tokenizer(&self) -> std::sync::Arc<dyn serve::Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), serve::EngineError> {
        self.inner.prefill(ids)?;
        let start = self.held();
        let end = start + ids.len();
        if let Some(tail) = self.tail
            && end > start + tail + 1
        {
            self.holes.push((start, end - tail));
        }
        self.hold(end);
        Ok(())
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, serve::EngineError> {
        let g = self.inner.next(last, out)?;
        self.hold(self.held() + 1);
        Ok(g)
    }
    fn reset(&mut self) -> Result<(), serve::EngineError> {
        self.hold(0);
        self.holes.clear();
        self.inner.reset()
    }
    fn keepable(&self, n: usize) -> usize {
        let k = self.rule(n);
        self.asked.lock().expect("log").asked.push((n, k));
        k
    }
    fn cut(&mut self, n: usize) -> Result<(), serve::EngineError> {
        if self.rule(n) != n {
            return Err(serve::EngineError(format!(
                "cut to {n} of {}: not a kept point",
                self.held()
            )));
        }
        self.hold(n);
        self.holes.retain(|&(start, _)| start < n);
        self.inner.cut(n)
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
    /// A restored cache has no prompt call of this engine's life: no hole.
    fn restore_state(
        &mut self,
        input: &mut dyn std::io::Read,
    ) -> Result<serve::SavedState, serve::StateError> {
        let s = self.inner.restore_state(input)?;
        self.hold(s.n_tokens);
        self.holes.clear();
        Ok(s)
    }
}

/// A server on a [`Spy`] with room for 4096 positions, and its log.
fn spy(tail: Option<usize>) -> (std::net::SocketAddr, Asked) {
    let asked = Asked::default();
    let addr = common::start_with(Box::new(Spy {
        inner: serve::MockEngine::new(4096),
        tail,
        holes: Vec::new(),
        asked: asked.clone(),
    }));
    (addr, asked)
}

fn shared(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap_or_else(|| panic!("not an id array: {v}"))
        .iter()
        .map(|t| u32::try_from(t.as_u64().expect("an id")).expect("a u32 id"))
        .collect()
}

/// What one warm request kept, against the engine it ran on.
struct Kept {
    /// The prefix it shared with what the most recent request left: the least
    /// the server may ask the engine to keep.
    last: usize,
    /// The longest it shared with what any earlier request left: the most.
    bound: usize,
    /// The `n` of the request's last `keepable`, 0 when it asked none.
    asked: usize,
    cache_n: usize,
}

/// Every id sequence a request left in the server's cache (its prompt and all
/// but its last generated id), and what the requests since the last check asked.
struct Ledger {
    helds: Vec<Vec<u32>>,
    asked: Asked,
}

impl Ledger {
    fn new(asked: Asked) -> Ledger {
        asked.lock().expect("log").asked.clear();
        Ledger {
            helds: Vec::new(),
            asked,
        }
    }

    /// Checks the warm reply `v` to a prompt of `p` against the engine's answers
    /// to the request's `keepable` calls and against the positions it holds
    /// after it, and books what the request left: `p` and `generated` less its
    /// last id.
    fn check(&mut self, v: &Value, p: &[u32], generated: &[u32]) -> Kept {
        let top = p.len() - 1;
        let last = self.helds.last().map_or(0, |h| shared(h, p)).min(top);
        let bound = self
            .helds
            .iter()
            .map(|h| shared(h, p))
            .max()
            .unwrap_or(0)
            .min(top);
        let (asked, engine_held) = {
            let mut log = self.asked.lock().expect("log");
            (std::mem::take(&mut log.asked), log.held)
        };
        let t = &v["timings"];
        let cache_n = usize::try_from(t["cache_n"].as_u64().expect("cache_n")).expect("usize");
        let at = format!("asked {asked:?}, last {last}, bound {bound}: {v}");
        let (n, answer) = asked.last().copied().unwrap_or((0, 0));
        assert!(
            last == 0 || !asked.is_empty(),
            "a prompt sharing {last} ids never asked the engine: {at}"
        );
        assert_eq!(
            cache_n,
            answer.min(n),
            "cache_n is the engine's answer: {at}"
        );
        assert!(last <= n && n <= bound, "the ask: {at}");
        assert_eq!(
            t["prompt_n"].as_u64().expect("prompt_n") + cache_n as u64,
            p.len() as u64,
            "{at}"
        );
        let mut held = p.to_vec();
        held.extend(&generated[..generated.len().saturating_sub(1)]);
        assert_eq!(
            engine_held,
            held.len(),
            "the engine holds what the server booked: {at}"
        );
        self.helds.push(held);
        Kept {
            last,
            bound,
            asked: n,
            cache_n,
        }
    }
}

/// Greedy `/completion` of `p` on a fresh server: the reference a warm
/// request's continuation must equal, id for id.
fn fresh(p: &[u32], extra: &Value) -> Value {
    let mut b = json!({"prompt": p, "temperature": 0, "return_tokens": true});
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e.clone());
    }
    post(start(4096), "/completion", &b).json()
}

/// A chat as a client drives it, turn by turn: each turn sends the earlier
/// messages, the assistant's `content` as answered (its reasoning left out) and
/// the next user message.
struct Chat {
    addr: std::net::SocketAddr,
    ledger: Ledger,
    convs: Vec<Vec<Value>>,
    /// Request fields every turn carries besides `messages`.
    extra: Value,
}

impl Chat {
    fn new(addr: std::net::SocketAddr, asked: Asked, extra: Value) -> Chat {
        Chat {
            addr,
            ledger: Ledger::new(asked),
            convs: Vec::new(),
            extra,
        }
    }

    /// Opens a conversation with `system` (none when empty); returns its index.
    fn open(&mut self, system: &str) -> usize {
        let m = if system.is_empty() {
            Vec::new()
        } else {
            vec![json!({"role": "system", "content": system})]
        };
        self.convs.push(m);
        self.convs.len() - 1
    }

    /// Sends conversation `c`'s next turn and checks it: the reuse against the
    /// engine's answer ([`Ledger::check`]), and the reply against a fresh
    /// server's greedy ids of the same rendered prompt. Returns the rendered ids
    /// and what the turn kept.
    fn turn(&mut self, c: usize, user: &str) -> (Vec<u32>, Kept) {
        use serve::Tokenizer;
        self.convs[c].push(json!({"role": "user", "content": user}));
        let mut body = json!({"messages": self.convs[c], "temperature": 0});
        if let (Value::Object(b), Value::Object(e)) = (&mut body, &self.extra) {
            b.extend(e.clone());
        }
        let text = post(self.addr, "/apply-template", &body).json()["prompt"]
            .as_str()
            .expect("a rendered prompt")
            .to_owned();
        let p = serve::MockTokenizer.encode(&text);
        let warm = post(self.addr, "/v1/chat/completions", &body).json();
        let mut limits = json!({});
        for k in ["ignore_eos", "max_tokens"] {
            if let Some(v) = body.get(k) {
                limits[if k == "max_tokens" { "n_predict" } else { k }] = v.clone();
            }
        }
        let f = fresh(&p, &limits);
        let g = ids(&f["tokens"]);
        let kept = self.ledger.check(&warm, &p, &g);
        let m = &warm["choices"][0]["message"];
        let content = m["content"].as_str().expect("content").to_owned();
        let reasoning = m["reasoning_content"].as_str().unwrap_or("");
        let raw = f["content"].as_str().expect("fresh content");
        let thinking = text.ends_with("<think>");
        let same = if thinking {
            raw == format!("{reasoning}</think>{content}")
                || (content.is_empty() && raw == reasoning)
        } else {
            reasoning.is_empty() && raw == content
        };
        assert!(same, "warm {warm}\nfresh {f}");
        assert_eq!(warm["usage"]["completion_tokens"], json!(g.len()), "{warm}");
        assert_eq!(warm["usage"]["prompt_tokens"], json!(p.len()), "{warm}");
        assert_eq!(
            warm["usage"]["prompt_tokens_details"]["cached_tokens"],
            json!(kept.cache_n),
            "{warm}"
        );
        self.convs[c].push(json!({"role": "assistant", "content": content}));
        (p, kept)
    }
}

/// The user message whose `</think>` the mock's reply follows, up to the end
/// of generation: with `tag` the reply is `tag` (after the chat's `</think>`).
fn reply_of(lead: &str, tag: &str) -> String {
    format!("{lead} </think>{tag}<｜end▁of▁sentence｜>")
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_thinking_turn_keeps_the_prompt_to_its_think() {
    let (addr, asked) = spy(None);
    let mut chat = Chat::new(
        addr,
        asked,
        json!({"chat_template_kwargs": {"thinking": true}, "max_tokens": 24}),
    );
    let c = chat.open("");
    // The mock thinks "ok" and answers "fine": the prompt ends in `<think>`, and
    // the user message's `<think>` is followed by that.
    let (p1, _) = chat.turn(c, "<think>ok</think>fine<｜end▁of▁sentence｜>");
    let (_, k2) = chat.turn(c, "more");
    // The recorded turn opens `</think>` where the generation prompt had
    // `<think>`: the two prompts share all of turn 1's but its last id.
    assert_eq!(k2.last, p1.len() - 1);
    assert_eq!(k2.cache_n, p1.len() - 1);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_every_later_turn_keeps_the_earlier_turns() {
    for tail in [None, Some(4)] {
        let (addr, asked) = spy(tail);
        let mut chat = Chat::new(addr, asked, json!({"max_tokens": 24}));
        let c = chat.open("You answer with three letters.");
        let mut prev = chat.turn(c, &reply_of("one", "xyz")).0;
        for (lead, tag) in [("two", "uvw"), ("three", "rst"), ("four", "opq")] {
            let (p, k) = chat.turn(c, &reply_of(lead, tag));
            // Append-only: the turn shares the previous prompt and its reply,
            // and every one of them is kept.
            assert!(k.last > prev.len() + 2, "tail {tail:?} {lead}");
            assert_eq!(k.cache_n, k.last, "tail {tail:?} {lead}");
            prev = p;
        }
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_reply_past_the_window_keeps_the_prompt_to_its_think() {
    let (addr, asked) = spy(Some(4));
    let mut chat = Chat::new(
        addr,
        asked,
        json!({"chat_template_kwargs": {"thinking": true}, "max_tokens": 160, "ignore_eos": true}),
    );
    let c = chat.open("");
    let (p1, _) = chat.turn(c, "<think>ok</think>fine<｜end▁of▁sentence｜>");
    let (_, k2) = chat.turn(c, "more");
    assert_eq!(k2.last, p1.len() - 1);
    assert_eq!(k2.cache_n, p1.len() - 1, "the cut is at a decode position");
}

/// A server whose every generation is the never-closing script
/// `abcdefghijkl` then EOS (the mock's vocabulary: one id a byte, the
/// specials single ids).
fn think_script() -> std::net::SocketAddr {
    common::start_with(Box::new(serve::ScriptedEngine::new(4096, "abcdefghijkl")))
}

/// A thinking-on chat of one user turn — the rendered prompt ends in
/// `<think>` — with `extra` beside it.
fn think_chat(extra: Value) -> Value {
    let mut b = json!({
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"thinking": true},
    });
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e.clone());
    }
    b
}

/// A never-closing script with `reasoning_budget` 4: the script's first four
/// tokens are the reasoning, the close id is forced as the fifth taken token
/// (the engine answer it displaces never appears), and the split consumes the
/// tag — `</think>` shows in neither field.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_budget_caps_the_span() {
    let addr = think_script();
    let r = post(
        addr,
        "/v1/chat/completions",
        &think_chat(json!({"reasoning_budget": 4})),
    )
    .json();
    let m = &r["choices"][0]["message"];
    let (reasoning, content) = (
        m["reasoning_content"].as_str().unwrap_or(""),
        m["content"].as_str().unwrap_or(""),
    );
    assert_eq!(reasoning, "abcd", "{r}");
    // The script's `e` is the answer the arming step displaced; the content
    // resumes at `f`.
    assert_eq!(content, "fghijkl", "{r}");
    assert!(
        !reasoning.contains("</think>") && !content.contains("</think>"),
        "{r}"
    );
    assert_eq!(r["usage"]["completion_tokens"], json!(13), "{r}");
    assert_eq!(r["choices"][0]["finish_reason"], json!("stop"), "{r}");
}

/// `reasoning_budget` 0 closes the span before any model token: no reasoning,
/// and the content starts at the script's second token — the first is the
/// answer the force displaced at the prompt's step.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_budget_zero_closes_at_once() {
    let addr = think_script();
    let r = post(
        addr,
        "/v1/chat/completions",
        &think_chat(json!({"reasoning_budget": 0})),
    )
    .json();
    let m = &r["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "{r}");
    assert_eq!(m["content"], json!("bcdefghijkl"), "{r}");
    assert_eq!(r["usage"]["completion_tokens"], json!(13), "{r}");
}

/// Absent, `null` and `-1` are the one unrestricted spelling: identical
/// replies, the span still open at the end keeping everything in reasoning.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_budget_absent_null_and_minus_one_are_unrestricted() {
    let addr = think_script();
    let free = post(addr, "/v1/chat/completions", &think_chat(json!({}))).json();
    for budget in [Value::Null, json!(-1)] {
        let r = post(
            addr,
            "/v1/chat/completions",
            &think_chat(json!({ "reasoning_budget": budget })),
        )
        .json();
        assert_eq!(
            r["choices"][0]["message"], free["choices"][0]["message"],
            "budget {budget}: {r}"
        );
        assert_eq!(
            r["usage"]["completion_tokens"], free["usage"]["completion_tokens"],
            "budget {budget}: {r}"
        );
    }
    let m = &free["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], json!("abcdefghijkl"), "{free}");
    assert_eq!(m["content"], json!(""), "{free}");
}

/// A value the field does not take is a 400 naming it (an integer under -1)
/// or its type (a non-integer number, a string, a boolean), on both
/// endpoints; `-1` and an integer-valued float serve.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_budget_parse_errors_are_refused() {
    let addr = start(4096);
    for (value, named) in [
        (json!(-2), "-2"),
        (json!(3.5), "a number"),
        (json!("8"), "a string"),
        (json!(true), "a boolean"),
    ] {
        for path in ["/completion", "/v1/chat/completions"] {
            let mut b = chat_body(json!({}));
            b["reasoning_budget"] = value.clone();
            let r = post(addr, path, &b);
            assert_eq!(r.status, 400, "{path} {value}: {}", r.body);
            let e = &r.json()["error"];
            assert_eq!(e["type"], "invalid_request_error", "{path} {value}: {e}");
            assert!(
                e["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("reasoning_budget") && m.contains(named)),
                "{path} {value}: the message must name the field and {named}: {e}"
            );
        }
    }
    for value in [json!(-1), json!(8.0)] {
        let mut b = chat_body(json!({}));
        b["prompt"] = json!("ab");
        b["reasoning_budget"] = value.clone();
        let r = post(addr, "/completion", &b);
        assert_eq!(r.status, 200, "{value}: {}", r.body);
    }
}

/// A prompt the template already closed (thinking off) silently ignores the
/// budget, as llama-server ignores the flag when thinking is off by other
/// means: no close id forced, the whole script the content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_budget_ignored_when_the_span_is_closed() {
    let addr = think_script();
    let r = post(
        addr,
        "/v1/chat/completions",
        &json!({
            "messages": [{"role": "user", "content": "q"}],
            "temperature": 0,
            "max_tokens": 32,
            "reasoning_budget": 0,
        }),
    )
    .json();
    let m = &r["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "{r}");
    assert_eq!(m["content"], json!("abcdefghijkl"), "{r}");
    assert_eq!(r["usage"]["completion_tokens"], json!(13), "{r}");
}

/// `/completion` carries the budget too, its prompt a string or an id array
/// (the array decoded for the span check): the raw reply shows the forced
/// close in place, the third taken token.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_budget_on_a_completion_prompt() {
    use serve::{MockTokenizer, Tokenizer};
    let addr = think_script();
    let close = MockTokenizer.encode("</think>");
    assert_eq!(close.len(), 1, "the mock's vocabulary closes in one id");
    for prompt in [json!("xy<think>"), json!(MockTokenizer.encode("xy<think>"))] {
        let r = post(
            addr,
            "/completion",
            &json!({
                "prompt": prompt, "temperature": 0, "n_predict": 32,
                "reasoning_budget": 2, "return_tokens": true,
            }),
        )
        .json();
        let at = format!("prompt {prompt}: {r}");
        assert_eq!(r["content"], json!("ab</think>defghijkl"), "{at}");
        let tokens = ids(&r["tokens"]);
        assert_eq!(tokens[2], close[0], "{at}");
        assert_eq!(tokens.len(), 13, "{at}");
        assert_eq!(r["tokens_predicted"], json!(13), "{at}");
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_divergence_inside_a_long_prompt_call_keeps_what_the_engine_grants() {
    use serve::Tokenizer;
    let first: String = (0..40).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
    let second = format!("{}{}", &first[..20], "zzzz");
    // Inside a first call: the last request asks for the 20 ids it shares; a
    // prompt call keeps only its start and its last four positions under the
    // hole rule, all of it without. Inside a second call (from position 10,
    // where the first request's cache ended): the start is 10.
    let later = format!("{}{}", &first[..30], "zzzz");
    let cases: [(&[&str], usize, [usize; 2]); 2] = [
        (&[&first, &second], 20, [20, 0]),
        (&[&first[..10], &first, &later], 30, [30, 10]),
    ];
    for (prompts, asks, want) in cases {
        for (tail, want) in [(None, want[0]), (Some(4), want[1])] {
            let (addr, asked) = spy(tail);
            let mut ledger = Ledger::new(asked);
            let mut k = None;
            for prompt in prompts {
                let p = serve::MockTokenizer.encode(prompt);
                let body = json!({"prompt": prompt, "n_predict": 6, "temperature": 0, "return_tokens": true});
                let warm = post(addr, "/completion", &body).json();
                let f = fresh(&p, &json!({"n_predict": 6}));
                assert_eq!(warm["tokens"], f["tokens"], "tail {tail:?}: {warm}\n{f}");
                k = Some(ledger.check(&warm, &p, &ids(&warm["tokens"])));
            }
            let k = k.expect("a request");
            assert_eq!(
                (k.asked, k.cache_n),
                (asks, want),
                "tail {tail:?} {prompts:?}"
            );
        }
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_an_interleaved_conversation_keeps_what_the_engine_grants() {
    let (addr, asked) = spy(None);
    let mut chat = Chat::new(addr, asked, json!({"max_tokens": 24}));
    let a = chat.open("Conversation A.");
    let b = chat.open("Another conversation, B.");
    let (pa, _) = chat.turn(a, &reply_of("a1", "xyz"));
    chat.turn(b, &reply_of("b1", "uvw"));
    let (_, k) = chat.turn(a, &reply_of("a2", "rst"));
    // A's second turn shares all of A's first turn with what A left, and much
    // less with what B left since; the server asks between the two.
    assert!(k.bound > pa.len(), "bound {} of {}", k.bound, pa.len());
    assert!(k.last < k.bound, "last {} bound {}", k.last, k.bound);
    assert!(k.cache_n <= k.bound);
}

/// Prompts whose greedy mock runs are long, short and stopped by EOS.
const DRAFT_PROMPTS: [&str; 3] = [
    "abcabcabcab",
    "the cat sat on the mat and the cat sat on the ",
    "xyz uvw xyz uv",
];

/// A greedy `/completion` of `prompt`: its ids and its reply.
fn drafted_ids(
    addr: std::net::SocketAddr,
    prompt: &Value,
    n: usize,
    cache: bool,
) -> (Vec<u64>, Value) {
    let v = post(
        addr,
        "/completion",
        &json!({"prompt": prompt, "n_predict": n, "temperature": 0, "return_tokens": true,
                "cache_prompt": cache}),
    )
    .json();
    let ids = v["tokens"]
        .as_array()
        .unwrap_or_else(|| panic!("no tokens: {v}"))
        .iter()
        .map(|t| t.as_u64().expect("an id"))
        .collect();
    (ids, v)
}

/// A drafting engine's greedy ids are the plain engine's, each prompt from a
/// reset cache and again as a continuation that keeps all but its last id;
/// `timings` carry the draft's counts, a part of its proposals kept, and
/// `/metrics` their sums.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_draft_gives_the_plain_ids() {
    let plain = start(4096);
    let drafted = common::start_with(Box::new(serve::DraftMock::new(4096)));
    let (mut proposed, mut accepted) = (0, 0);
    for p in DRAFT_PROMPTS {
        let (want, _) = drafted_ids(plain, &json!(p), 40, false);
        let (got, v) = drafted_ids(drafted, &json!(p), 40, false);
        assert_eq!(got, want, "{p:?}: {v}");
        let t = &v["timings"];
        let (n, a) = (t["draft_n"].as_u64(), t["draft_n_accepted"].as_u64());
        let (n, a) = n
            .zip(a)
            .unwrap_or_else(|| panic!("{p:?}: no draft counts: {t}"));
        assert!(0 < a && a < n, "{p:?}: {t}");
        assert_eq!(t["predicted_n"], json!(got.len()), "{t}");
        (proposed, accepted) = (proposed + n, accepted + a);
        let prompt_ids =
            post(drafted, "/tokenize", &json!({"content": p})).json()["tokens"].clone();
        let mut cont: Vec<Value> = prompt_ids.as_array().expect("ids").clone();
        cont.extend(got.iter().map(|&id| json!(id)));
        let (fresh, _) = drafted_ids(plain, &json!(cont), 16, false);
        let (warm, v) = drafted_ids(drafted, &json!(cont), 16, true);
        assert_eq!(warm, fresh, "{p:?} continued: {v}");
        assert_eq!(v["timings"]["cache_n"], json!(cont.len() - 1), "{v}");
        let t = &v["timings"];
        (proposed, accepted) = (
            proposed + t["draft_n"].as_u64().unwrap_or(0),
            accepted + t["draft_n_accepted"].as_u64().unwrap_or(0),
        );
    }
    let m = get(drafted, "/metrics").body;
    assert_eq!(
        metric(&m, "spec_decode_num_draft_tokens_total"),
        proposed as f64
    );
    assert_eq!(
        metric(&m, "spec_decode_num_accepted_tokens_total"),
        accepted as f64
    );
    let v = drafted_ids(plain, &json!(DRAFT_PROMPTS[0]), 8, false).1;
    assert!(
        v["timings"].get("draft_n").is_none(),
        "a plain engine drafts nothing: {v}"
    );
    let e = get(drafted, "/props").json()["engine"].clone();
    assert_eq!(
        e["draft"],
        json!({"model": "mock", "n_max": 1, "kind": "mock"}),
        "{e}"
    );
}

/// Near the context's end a pass that would pass it is not run: the positions
/// left take one step each, so the drafting engine's ids end where the plain
/// engine's do. Six contexts in a row put the end at every phase of the
/// mock's three-pass draft.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_draft_stops_at_the_plain_context_end() {
    for ctx in 40..46 {
        let plain = start(ctx);
        let drafted = common::start_with(Box::new(serve::DraftMock::new(ctx)));
        let p = json!(DRAFT_PROMPTS[0]);
        let (want, w) = drafted_ids(plain, &p, 100, false);
        let (got, v) = drafted_ids(drafted, &p, 100, false);
        assert_eq!(got, want, "ctx {ctx}: {v}");
        assert_eq!(w["truncated"], true, "{w}");
        assert_eq!(v["truncated"], true, "{v}");
    }
}

/// A `/completion` of `body` with `return_tokens`, which must be served: its
/// ids and its reply.
fn served_ids(addr: std::net::SocketAddr, body: &Value) -> (Vec<u64>, Value) {
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

/// A drafting engine serves what needs the logits row — sampling, and a banned
/// end-of-generation id — through plain steps, llama-server's default
/// temperature included: each request's ids are the plain engine's, and it
/// drafts nothing (no draft counts in its `timings`, `/metrics`' draft sums
/// unmoved); a greedy request after them still drafts. The sampled requests
/// branch: some seed's ids are not the greedy ones. Mutant: the refusal of a
/// sampled or id-banning request under a draft restored (a 400).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_draft_serves_what_needs_the_logits() {
    let plain = start(4096);
    let drafted = common::start_with(Box::new(serve::DraftMock::new(4096).without_sampled()));
    // `a` has four followers here, so a sampler at T = 1.5 branches; min_p
    // 0.01 keeps only those followers (the rest sit 12 logits down).
    let prompt = "abacadaeabacada";
    let sampled = |seed: u64| {
        json!({"prompt": prompt, "n_predict": 24, "temperature": 1.5, "top_k": 0,
               "top_p": 1.0, "min_p": 0.01, "seed": seed})
    };
    let no_draft = |v: &Value| {
        assert!(
            v["timings"].get("draft_n").is_none(),
            "a stepped request drafts nothing: {v}"
        );
    };
    let (greedy, _) = served_ids(
        plain,
        &json!({"prompt": prompt, "n_predict": 24, "temperature": 0}),
    );
    let mut branched = false;
    for seed in 1..=4 {
        let (want, _) = served_ids(plain, &sampled(seed));
        let (got, v) = served_ids(drafted, &sampled(seed));
        assert_eq!(got, want, "seed {seed}: {v}");
        no_draft(&v);
        branched |= got != greedy;
    }
    assert!(branched, "no seed sampled off the greedy ids {greedy:?}");
    let eos = json!({"prompt": "xyz", "ignore_eos": true, "n_predict": 4, "temperature": 0});
    let (want, _) = served_ids(plain, &eos);
    let (got, v) = served_ids(drafted, &eos);
    assert_eq!(got.len(), 4, "{v}");
    assert_eq!(got, want, "{v}");
    no_draft(&v);
    let chat = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                      "seed": 7});
    let content = |addr| {
        let r = post(addr, "/v1/chat/completions", &chat);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        no_draft(&v);
        v["choices"][0]["message"]["content"].clone()
    };
    assert_eq!(content(drafted), content(plain));
    let m = get(drafted, "/metrics").body;
    assert_eq!(metric(&m, "spec_decode_num_draft_tokens_total"), 0.0);
    assert_eq!(metric(&m, "spec_decode_num_drafts_total"), 0.0);
    let greedy_body = json!({"prompt": "abcabcabcab", "n_predict": 12, "temperature": 0});
    let (want, _) = served_ids(plain, &greedy_body);
    let (got, v) = served_ids(drafted, &greedy_body);
    assert_eq!(got, want, "{v}");
    assert!(
        v["timings"]["draft_n"].as_u64().is_some_and(|n| n > 0),
        "a greedy request drafts: {v}"
    );
}

/// `POST /residency/reset`: an engine with no residency is a 501; one with a
/// residency answers the reset's report, and only this call resets it — a
/// completion before or after it does not.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_residency_reset_is_the_one_explicit_call() {
    let plain = start(64);
    let r = call(plain, "POST", "/residency/reset", None);
    assert_eq!(r.status, 501, "{}", r.body);

    let resets = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = || resets.load(std::sync::atomic::Ordering::SeqCst);
    let addr = common::start_with(Box::new(serve::MockEngine::with_residency(
        64,
        std::sync::Arc::clone(&resets),
    )));
    let body = json!({"prompt": "abcabc", "n_predict": 3, "temperature": 0});
    assert_eq!(post(addr, "/completion", &body).status, 200);
    assert_eq!(count(), 0, "a completion reset the residency");
    let r = call(addr, "POST", "/residency/reset", None);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    for k in ["cancelled", "copies", "diff", "dropped_bytes"] {
        assert!(v[k].is_u64(), "{k} in {v}");
    }
    assert!(v["timings"]["reset_ms"].is_f64(), "{v}");
    assert_eq!(count(), 1);
    assert_eq!(post(addr, "/completion", &body).status, 200);
    assert_eq!(count(), 1, "a completion reset the residency");
    // A prompt that shares no prefix: the cache misses and the engine resets.
    let miss = json!({"prompt": "zyxzyx", "n_predict": 3, "temperature": 0});
    assert_eq!(post(addr, "/completion", &miss).status, 200);
    assert_eq!(count(), 1, "a cache miss reset the residency");
}
