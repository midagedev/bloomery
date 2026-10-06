//! Gate: Anthropic's Messages API (`POST /v1/messages`, `POST /v1/messages/count_tokens`) over
//! the mock engine. Each test ports the cases of llama-server's
//! `tools/server/tests/unit/test_compat_anthropic.py` it names, held to exact values where the
//! Python cases accept any model output.
//!
//! The mock's greedy output for [`turns`] is `abc` then EOS (`tests/serve.rs` derives it from the
//! mock's bigram rule): 15 prompt ids, 4 generated. A scripted engine stands in for a reasoning
//! model and for a tool-calling one.

mod common;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use common::{Reply, call, get, post, start, start_with};
use serde_json::{Map, Value, json};
use serve::{MockEngine, ScriptedEngine};

/// The three-turn conversation the mock answers `abc`, with `extra` beside it.
fn turns(extra: Value) -> Value {
    with(
        json!({
            "model": "claude-test",
            "max_tokens": 32,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "abc"},
                {"role": "user", "content": "hi"},
            ],
        }),
        extra,
    )
}

/// `b` with the fields of `extra` set over its own.
fn with(mut b: Value, extra: Value) -> Value {
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e);
    }
    b
}

/// The data objects of an Anthropic stream, in order. Every frame must be an `event:` line and a
/// `data:` line whose JSON's `type` is the event's name; nothing else may follow the frames.
fn frames(r: &Reply) -> Vec<Value> {
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(
        r.header("content-type")
            .is_some_and(|c| c.starts_with("text/event-stream")),
        "{:?}",
        r.headers
    );
    r.body
        .split("\n\n")
        .filter(|f| !f.is_empty())
        .map(|f| {
            let (ev, data) = f
                .split_once('\n')
                .unwrap_or_else(|| panic!("a frame of one line: {f:?}"));
            let ev = ev
                .strip_prefix("event: ")
                .unwrap_or_else(|| panic!("no event line: {f:?}"));
            let data = data
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("no data line: {f:?}"));
            let v: Value = serde_json::from_str(data)
                .unwrap_or_else(|e| panic!("data is not JSON ({e}): {data}"));
            assert_eq!(v["type"], ev, "the event's name is its data's type: {f:?}");
            v
        })
        .collect()
}

fn types(frames: &[Value]) -> Vec<&str> {
    frames
        .iter()
        .map(|f| f["type"].as_str().expect("a type"))
        .collect()
}

/// The text a stream's `kind` deltas carry under `field`, joined.
fn joined(frames: &[Value], kind: &str, field: &str) -> String {
    frames
        .iter()
        .filter(|f| f["type"] == "content_block_delta" && f["delta"]["type"] == kind)
        .map(|f| f["delta"][field].as_str().expect("delta text"))
        .collect()
}

/// The expected event types: `message_start`, then per block its start, `n` deltas and its stop,
/// then `message_delta` and `message_stop`.
fn expected_types(blocks: &[usize]) -> Vec<&'static str> {
    let mut t = vec!["message_start"];
    for &n in blocks {
        t.push("content_block_start");
        t.extend(std::iter::repeat_n("content_block_delta", n));
        t.push("content_block_stop");
    }
    t.extend(["message_delta", "message_stop"]);
    t
}

/// How many deltas each block of a stream carries, in block order.
fn deltas_per_block(frames: &[Value]) -> Vec<usize> {
    let mut n = Vec::new();
    for f in frames {
        match f["type"].as_str() {
            Some("content_block_start") => n.push(0),
            Some("content_block_delta") => *n.last_mut().expect("a delta inside a block") += 1,
            _ => {}
        }
    }
    n
}

/// Asserts an error answer: `status`, and exactly Anthropic's envelope around the OpenAI path's
/// object — `code` the status, `type` `kind`, a `message` holding `part`.
fn assert_envelope(r: &Reply, status: u16, kind: &str, part: &str) {
    assert_eq!(r.status, status, "{}", r.body);
    let v = r.json();
    let keys: Vec<&String> = v.as_object().expect("an object").keys().collect();
    assert_eq!(keys, ["type", "error"], "{v}");
    assert_eq!(v["type"], "error", "{v}");
    let e = &v["error"];
    let mut ekeys: Vec<&String> = e.as_object().expect("an error object").keys().collect();
    ekeys.sort();
    assert_eq!(ekeys, ["code", "message", "type"], "{v}");
    assert_eq!(e["code"], status, "{v}");
    assert_eq!(e["type"], kind, "{v}");
    assert!(
        e["message"].as_str().is_some_and(|m| m.contains(part)),
        "the message must hold {part:?}: {v}"
    );
}

/// One `POST` over HTTP/1.0 carrying Anthropic's headers, as the SDK and Claude Code send them.
fn post_as_anthropic_client(addr: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(60)))
        .expect("timeout");
    let body = body.to_string();
    let req = format!(
        "POST {path} HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         x-api-key: sk-ant-not-checked\r\nanthropic-version: 2023-06-01\r\n\
         anthropic-beta: interleaved-thinking-2025-05-14\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("write");
    let mut raw = String::new();
    s.read_to_string(&mut raw).expect("read");
    let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .expect("status line");
    let v = serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}): {body}"));
    (status, v)
}

/// The rendered prompt of the last request the server's one slot ran.
fn last_prompt(addr: SocketAddr) -> Value {
    get(addr, "/slots").json()[0]["prompt"].clone()
}

/// Ports `test_anthropic_messages_basic` and `test_anthropic_vs_openai_different_response_format`:
/// the message object holds exactly llama-server's keys in its order — no `timings` — the mock's
/// greedy text as one text block, `end_turn`, and a `usage` that splits the OpenAI path's prompt
/// count of the same request into the ids the cache kept and the rest. The request carries
/// Anthropic's headers (`x-api-key`, `anthropic-version`, `anthropic-beta`); the server checks no key.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_messages_basic() {
    let addr = start(4096);
    let (status, v) = post_as_anthropic_client(addr, "/v1/messages", &turns(json!({})));
    assert_eq!(status, 200, "{v}");
    let keys: Vec<&String> = v.as_object().expect("an object").keys().collect();
    assert_eq!(
        keys,
        [
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "usage"
        ],
        "{v}"
    );
    let id = v["id"].as_str().expect("an id");
    assert!(
        id.len() == 4 + 32
            && id.starts_with("msg_")
            && id[4..].bytes().all(|b| b.is_ascii_hexdigit()),
        "{v}"
    );
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "claude-test");
    assert_eq!(v["content"], json!([{"type": "text", "text": "abc"}]));
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["stop_sequence"], Value::Null);
    assert_eq!(
        v["usage"],
        json!({"cache_read_input_tokens": 0, "input_tokens": 15, "output_tokens": 4})
    );
    let o = post(addr, "/v1/chat/completions", &turns(json!({}))).json();
    assert_eq!(o["object"], "chat.completion", "{o}");
    assert_eq!(o["choices"][0]["message"]["content"], "abc", "{o}");
    assert_eq!(o["usage"]["prompt_tokens"], 15, "{o}");
    assert_eq!(o["usage"]["completion_tokens"], 4, "{o}");
    let again = post(addr, "/v1/messages", &turns(json!({}))).json();
    assert_eq!(
        again["usage"],
        json!({"cache_read_input_tokens": 14, "input_tokens": 1, "output_tokens": 4}),
        "the slot keeps all but the prompt's last id"
    );
}

/// Ports `test_anthropic_messages_with_system`, `test_anthropic_messages_multipart_content`,
/// `test_anthropic_messages_conversation`, `test_anthropic_tool_result`,
/// `test_anthropic_tool_result_with_text`, `test_anthropic_tool_result_error`,
/// `test_anthropic_empty_messages`, `test_anthropic_metadata` and the conversion half of
/// `test_anthropic_thinking_history_in_template`: an Anthropic request renders the prompt its
/// OpenAI equivalent renders under llama-server's conversion — the system prompt (a string, or
/// text blocks joined with nothing between them) first, text and thinking and `tool_use` blocks as
/// one message's parts, `reasoning_content` and `tool_calls`, each `tool_result` a `tool` message
/// after its message (text blocks joined), an assistant message without content dropped, the
/// tools as functions; `cache_control`, `is_error`, a thinking block's `signature` and `metadata`
/// change nothing.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_request_conversion() {
    let addr = start(8192);
    let weather = json!({
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    });
    let cases = [
        (
            json!({
                "system": "Be brief.",
                "metadata": {"user_id": "u1"},
                "messages": [{"role": "user", "content": "hi"}],
            }),
            json!({"messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ]}),
        ),
        (
            json!({
                "system": [
                    {"type": "text", "text": "Be "},
                    {"type": "text", "text": "brief.", "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": " the answer?", "cache_control": {"type": "ephemeral"}},
                ]}],
            }),
            json!({"messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": " the answer?"},
                ]},
            ]}),
        ),
        (
            json!({"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "assistant"},
                {"role": "user", "content": "How are you?"},
            ]}),
            json!({"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]}),
        ),
        (
            json!({
                "tools": [{
                    "name": "get_weather",
                    "description": "Get the weather",
                    "input_schema": weather,
                    "cache_control": {"type": "ephemeral"},
                }],
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "t1", "name": "get_weather",
                         "input": {"location": "Paris"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Here are the results:"},
                        {"type": "tool_result", "tool_use_id": "t1", "is_error": true,
                         "content": "City not found"},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "t2", "name": "get_weather",
                         "input": {"location": "Lyon"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t2", "content": [
                            {"type": "text", "text": "Sunny, "},
                            {"type": "text", "text": "25°C"},
                        ]},
                    ]},
                ],
            }),
            json!({
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Get the weather", "parameters": weather,
                }}],
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Checking."}],
                     "tool_calls": [{"id": "t1", "type": "function", "function": {
                         "name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}}]},
                    {"role": "user", "content": [{"type": "text", "text": "Here are the results:"}]},
                    {"role": "tool", "tool_call_id": "t1", "content": "City not found"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "t2", "type": "function", "function": {
                         "name": "get_weather", "arguments": "{\"location\":\"Lyon\"}"}}]},
                    {"role": "tool", "tool_call_id": "t2", "content": "Sunny, 25°C"},
                ],
            }),
        ),
        (
            json!({
                "chat_template_kwargs": {"thinking": true},
                "messages": [
                    {"role": "user", "content": "Fix the bug"},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "Check the layout first.",
                         "signature": "sig"},
                        {"type": "tool_use", "id": "c1", "name": "list_files",
                         "input": {"path": "."}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": "main.py"},
                    ]},
                ],
            }),
            json!({
                "chat_template_kwargs": {"thinking": true},
                "messages": [
                    {"role": "user", "content": "Fix the bug"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "c1", "type": "function", "function": {
                         "name": "list_files", "arguments": "{\"path\":\".\"}"}}],
                     "reasoning_content": "Check the layout first."},
                    {"role": "tool", "tool_call_id": "c1", "content": "main.py"},
                ],
            }),
        ),
        (json!({"messages": []}), json!({"messages": []})),
    ];
    for (anthropic, openai) in cases {
        let r = post(
            addr,
            "/v1/messages",
            &with(
                anthropic.clone(),
                json!({"max_tokens": 1, "temperature": 0}),
            ),
        );
        assert_eq!(r.status, 200, "{anthropic}: {}", r.body);
        assert_eq!(r.json()["type"], "message", "{anthropic}");
        let want = post(addr, "/apply-template", &openai);
        assert_eq!(want.status, 200, "{openai}: {}", want.body);
        assert_eq!(
            last_prompt(addr),
            want.json()["prompt"],
            "{anthropic} renders as {openai}"
        );
    }
}

/// Ports `test_anthropic_messages_streaming`: the stream is exactly `message_start` (the message
/// with no content, `usage` the prompt's, `output_tokens` 0), one text block's start, its deltas and
/// its stop, `message_delta` (`end_turn`, the generated ids) and `message_stop`, each frame an
/// `event:` and a `data:` line; the deltas join to the non-stream text, and no frame carries
/// `timings` and no `[DONE]` follows.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_messages_streaming() {
    let addr = start(4096);
    let whole = post(addr, "/v1/messages", &turns(json!({}))).json();
    let r = post(addr, "/v1/messages", &turns(json!({"stream": true})));
    let f = frames(&r);
    assert_eq!(
        types(&f),
        expected_types(&deltas_per_block(&f)),
        "{}",
        r.body
    );
    assert_eq!(deltas_per_block(&f).len(), 1, "{}", r.body);
    let start = &f[0]["message"];
    let id = start["id"].as_str().expect("an id");
    assert!(id.starts_with("msg_") && id != whole["id"], "{start}");
    assert_eq!(
        *start,
        json!({
            "id": id, "type": "message", "role": "assistant", "content": [],
            "model": "claude-test", "stop_reason": null, "stop_sequence": null,
            "usage": {"cache_read_input_tokens": 14, "input_tokens": 1, "output_tokens": 0},
        })
    );
    assert_eq!(
        f[1],
        json!({"type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}})
    );
    for d in &f[2..f.len() - 3] {
        assert_eq!(d["index"], 0, "{d}");
        assert_eq!(d["delta"]["type"], "text_delta", "{d}");
    }
    assert_eq!(
        joined(&f, "text_delta", "text"),
        whole["content"][0]["text"].as_str().expect("text")
    );
    let n = f.len();
    assert_eq!(f[n - 3], json!({"type": "content_block_stop", "index": 0}));
    assert_eq!(
        f[n - 2],
        json!({"type": "message_delta",
               "delta": {"stop_reason": "end_turn", "stop_sequence": null},
               "usage": {"output_tokens": 4}})
    );
    assert_eq!(f[n - 1], json!({"type": "message_stop"}));
    assert!(
        !r.body.contains("timings") && !r.body.contains("[DONE]"),
        "{}",
        r.body
    );
}

/// Ports `test_anthropic_stop_sequences` and the `stop_reason` assertions of the Python cases:
/// `max_tokens` at the request's limit and at the context's end, `stop_sequence` with the sequence
/// that matched (cut from the text), `end_turn` at the end of generation, the same in the stream's
/// `message_delta`.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_stop_sequences() {
    let addr = start(4096);
    // The prompt is 15 ids: a context of 17 takes two generated ids, so the third stops it.
    let short = start(17);
    let cases = [
        (
            addr,
            json!({"max_tokens": 2}),
            "ab",
            "max_tokens",
            Value::Null,
            2,
        ),
        (short, json!({}), "abc", "max_tokens", Value::Null, 3),
        (
            addr,
            json!({"stop_sequences": ["bc"]}),
            "a",
            "stop_sequence",
            json!("bc"),
            3,
        ),
        (addr, json!({}), "abc", "end_turn", Value::Null, 4),
    ];
    for (at, extra, text, reason, sequence, output) in cases {
        let v = post(at, "/v1/messages", &turns(extra.clone())).json();
        assert_eq!(
            v["content"],
            json!([{"type": "text", "text": text}]),
            "{extra}: {v}"
        );
        assert_eq!(v["stop_reason"], reason, "{extra}: {v}");
        assert_eq!(v["stop_sequence"], sequence, "{extra}: {v}");
        assert_eq!(v["usage"]["output_tokens"], output, "{extra}: {v}");
        let f = frames(&post(
            at,
            "/v1/messages",
            &turns(with(extra.clone(), json!({"stream": true}))),
        ));
        assert_eq!(joined(&f, "text_delta", "text"), text, "{extra}");
        let delta = &f[f.len() - 2];
        assert_eq!(
            *delta,
            json!({"type": "message_delta",
                   "delta": {"stop_reason": reason, "stop_sequence": sequence},
                   "usage": {"output_tokens": output}}),
            "{extra}"
        );
    }
}

/// A server whose every generation is `plan</think>answer` then EOS: a reasoning model's span.
fn reasoning_model() -> SocketAddr {
    start_with(Box::new(ScriptedEngine::new(4096, "plan</think>answer")))
}

/// Ports `test_anthropic_thinking_with_reasoning_model` (both arms): with the template's thinking on
/// (the prompt ends in `<think>`) the span is a thinking block at index 0, its `signature` empty, and
/// the text a text block at index 1, in the message and in the stream — where the thinking block's
/// deltas end with an empty `signature_delta` and it stops before the text block starts, also when
/// one piece of the generation holds the end of the span and the start of the text.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_thinking_with_reasoning_model() {
    let addr = reasoning_model();
    let body = json!({
        "max_tokens": 64,
        "chat_template_kwargs": {"thinking": true},
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    });
    let v = post(addr, "/v1/messages", &body).json();
    assert_eq!(
        v["content"],
        json!([
            {"type": "thinking", "thinking": "plan", "signature": ""},
            {"type": "text", "text": "answer"},
        ]),
        "{v}"
    );
    assert_eq!(v["stop_reason"], "end_turn", "{v}");
    // The stop sequence holds `n</think>a` back until it fails to match, then releases it as
    // one piece that crosses the span's close.
    for extra in [
        json!({"stream": true}),
        json!({"stream": true, "stop_sequences": ["n</think>aQ"]}),
    ] {
        let r = post(addr, "/v1/messages", &with(body.clone(), extra));
        let f = frames(&r);
        let per_block = deltas_per_block(&f);
        assert_eq!(types(&f), expected_types(&per_block), "{}", r.body);
        assert_eq!(per_block.len(), 2, "{}", r.body);
        assert_eq!(
            f[1],
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "thinking", "thinking": ""}})
        );
        let thinking_end = 1 + per_block[0];
        assert_eq!(
            f[thinking_end],
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": ""}})
        );
        assert_eq!(
            f[thinking_end + 1],
            json!({"type": "content_block_stop", "index": 0})
        );
        assert_eq!(
            f[thinking_end + 2],
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "text", "text": ""}})
        );
        for d in &f[2..thinking_end] {
            assert_eq!(
                (&d["index"], &d["delta"]["type"]),
                (&json!(0), &json!("thinking_delta")),
                "{d}"
            );
        }
        for d in f.iter().skip(thinking_end + 3).take(per_block[1]) {
            assert_eq!(
                (&d["index"], &d["delta"]["type"]),
                (&json!(1), &json!("text_delta")),
                "{d}"
            );
        }
        assert_eq!(joined(&f, "thinking_delta", "thinking"), "plan");
        assert_eq!(joined(&f, "text_delta", "text"), "answer");
        assert_eq!(
            f[f.len() - 3],
            json!({"type": "content_block_stop", "index": 1})
        );
    }
}

/// Ports `test_anthropic_thinking`: `thinking` `enabled` with `budget_tokens` N caps the span as the
/// OpenAI path's `reasoning_budget` N does (the same thinking and text, here `abcd` and `fghijkl`);
/// `disabled`, `adaptive` and no `thinking` leave it unrestricted (the never-closing script all
/// thinking). `thinking` does not turn the template's thinking on: without the template's own
/// switch the prompt has closed the span, the budget is ignored and the whole script is text.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_thinking() {
    let addr = start_with(Box::new(ScriptedEngine::new(4096, "abcdefghijkl")));
    let ask = |thinking: Value, kwargs: bool| {
        let mut b = json!({
            "max_tokens": 32,
            "temperature": 0,
            "messages": [{"role": "user", "content": "q"}],
        });
        if kwargs {
            b["chat_template_kwargs"] = json!({"thinking": true});
        }
        if !thinking.is_null() {
            b["thinking"] = thinking;
        }
        let r = post(addr, "/v1/messages", &b);
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["content"].clone()
    };
    let capped = ask(json!({"type": "enabled", "budget_tokens": 4}), true);
    assert_eq!(
        capped,
        json!([
            {"type": "thinking", "thinking": "abcd", "signature": ""},
            {"type": "text", "text": "fghijkl"},
        ])
    );
    let o = post(
        addr,
        "/v1/chat/completions",
        &json!({
            "max_tokens": 32, "temperature": 0, "reasoning_budget": 4,
            "chat_template_kwargs": {"thinking": true},
            "messages": [{"role": "user", "content": "q"}],
        }),
    )
    .json();
    let m = &o["choices"][0]["message"];
    assert_eq!(
        (&m["reasoning_content"], &m["content"]),
        (&capped[0]["thinking"], &capped[1]["text"]),
        "{o}"
    );
    let free = json!([{"type": "thinking", "thinking": "abcdefghijkl", "signature": ""}]);
    for thinking in [
        Value::Null,
        json!({"type": "disabled"}),
        json!({"type": "adaptive"}),
    ] {
        assert_eq!(ask(thinking.clone(), true), free, "{thinking}");
    }
    assert_eq!(
        ask(json!({"type": "enabled", "budget_tokens": 4}), false),
        json!([{"type": "text", "text": "abcdefghijkl"}])
    );
}

/// The DSML markup of a text and two calls, as the V4.1 template teaches it.
const TWO_CALLS: &str = "Checking.\n\n<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"get_time\">\n\
<｜DSML｜parameter name=\"zone\" string=\"false\">1</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

/// Ports `test_anthropic_tool_use_basic`, `test_anthropic_tool_streaming` and
/// `test_anthropic_streaming_content_block_indices`: the calls the template's markup parses go out
/// as `tool_use` blocks after the text block — `toolu_<index>_<the message id's first 16>`, the
/// name, the input object — and `stop_reason` is `tool_use`; in the stream each call is a block of
/// its own at the next index, its start carrying the id, the name and an empty input, one
/// `input_json_delta` the whole input, each block stopped before the next starts. `tool_choice`
/// `none` leaves the markup unparsed, in the text.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_tool_use_basic() {
    let addr = start_with(Box::new(ScriptedEngine::new(4096, TWO_CALLS)));
    let body = json!({
        "max_tokens": 400,
        "tools": [
            {"name": "get_weather", "description": "Get the weather",
             "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}},
            {"name": "get_time", "input_schema": {"type": "object"}},
        ],
        "messages": [{"role": "user", "content": "Weather and time in Paris?"}],
    });
    let v = post(addr, "/v1/messages", &body).json();
    let nonce = &v["id"].as_str().expect("an id")[4..20];
    assert_eq!(
        v["content"],
        json!([
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": format!("toolu_0_{nonce}"), "name": "get_weather",
             "input": {"location": "Paris"}},
            {"type": "tool_use", "id": format!("toolu_1_{nonce}"), "name": "get_time",
             "input": {"zone": 1}},
        ]),
        "{v}"
    );
    assert_eq!(v["stop_reason"], "tool_use", "{v}");

    let r = post(
        addr,
        "/v1/messages",
        &with(body.clone(), json!({"stream": true})),
    );
    let f = frames(&r);
    let per_block = deltas_per_block(&f);
    assert_eq!(types(&f), expected_types(&per_block), "{}", r.body);
    assert_eq!(&per_block[1..], [1, 1], "{}", r.body);
    let nonce = &f[0]["message"]["id"].as_str().expect("an id")[4..20];
    let starts: Vec<&Value> = f
        .iter()
        .filter(|e| e["type"] == "content_block_start")
        .collect();
    let stops: Vec<&Value> = f
        .iter()
        .filter(|e| e["type"] == "content_block_stop")
        .collect();
    for (i, (a, b)) in starts.iter().zip(&stops).enumerate() {
        assert_eq!(
            (&a["index"], &b["index"]),
            (&json!(i), &json!(i)),
            "{}",
            r.body
        );
    }
    assert_eq!(
        starts[0]["content_block"],
        json!({"type": "text", "text": ""})
    );
    assert_eq!(joined(&f, "text_delta", "text"), "Checking.");
    let calls = [
        ("get_weather", r#"{"location":"Paris"}"#),
        ("get_time", r#"{"zone":1}"#),
    ];
    let json_deltas: Vec<&Value> = f
        .iter()
        .filter(|e| e["delta"]["type"] == "input_json_delta")
        .collect();
    for (i, (name, input)) in calls.iter().enumerate() {
        assert_eq!(
            starts[i + 1]["content_block"],
            json!({"type": "tool_use", "id": format!("toolu_{i}_{nonce}"),
                   "name": name, "input": {}})
        );
        assert_eq!(
            *json_deltas[i],
            json!({"type": "content_block_delta", "index": i + 1,
                   "delta": {"type": "input_json_delta", "partial_json": input}})
        );
    }
    assert_eq!(f[f.len() - 2]["delta"]["stop_reason"], "tool_use");

    let none = post(
        addr,
        "/v1/messages",
        &with(body, json!({"tool_choice": {"type": "none"}})),
    )
    .json();
    assert_eq!(
        none["content"],
        json!([{"type": "text", "text": TWO_CALLS}]),
        "{none}"
    );
    assert_eq!(none["stop_reason"], "end_turn", "{none}");
}

/// Ports `test_anthropic_count_tokens`, `test_anthropic_count_tokens_with_system`,
/// `test_anthropic_count_tokens_no_max_tokens` and `test_anthropic_thinking_history_in_count_tokens`:
/// `input_tokens`, the one key, is the ids of the prompt `/v1/messages` runs for the same request
/// (its `input_tokens` plus `cache_read_input_tokens`, and `/tokenize` of the prompt it rendered),
/// with no `max_tokens` asked; a system prompt and thinking in the history add their ids; what
/// `/v1/messages` refuses, it refuses.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_count_tokens() {
    let addr = start(4096);
    let count = |b: &Value| {
        let r = post(addr, "/v1/messages/count_tokens", b);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let keys: Vec<&String> = v.as_object().expect("an object").keys().collect();
        assert_eq!(keys, ["input_tokens"], "{v}");
        v["input_tokens"].as_u64().expect("a count")
    };
    let mut no_max = turns(json!({}));
    no_max
        .as_object_mut()
        .expect("an object")
        .remove("max_tokens");
    let n = count(&no_max);
    assert_eq!(n, 15);
    let u = post(addr, "/v1/messages", &turns(json!({}))).json()["usage"].clone();
    assert_eq!(
        u["input_tokens"].as_u64().expect("input")
            + u["cache_read_input_tokens"].as_u64().expect("cache"),
        n,
        "{u}"
    );
    let ids = post(
        addr,
        "/tokenize",
        &json!({"content": last_prompt(addr), "add_special": false}),
    )
    .json();
    assert_eq!(ids["tokens"].as_array().map(Vec::len), Some(15), "{ids}");
    let hi = json!({"messages": [{"role": "user", "content": "Hello"}]});
    let with_system = with(
        hi.clone(),
        json!({"system": "You are a helpful assistant."}),
    );
    assert!(count(&with_system) > count(&hi));
    let history = |thinking: bool| {
        let mut assistant = vec![json!({"type": "tool_use", "id": "c1", "name": "list_files",
                                        "input": {"path": "."}})];
        if thinking {
            assistant.insert(
                0,
                json!({"type": "thinking", "thinking": "Check the layout first."}),
            );
        }
        json!({
            "chat_template_kwargs": {"thinking": true},
            "tools": [{"name": "list_files", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "main.py"},
                ]},
            ],
        })
    };
    assert!(count(&history(true)) > count(&history(false)));
    let refused = with(
        hi,
        json!({"messages": [{"role": "user", "content": [{"type": "document"}]}]}),
    );
    assert_envelope(
        &post(addr, "/v1/messages/count_tokens", &refused),
        400,
        "invalid_request_error",
        "content block type \"document\"",
    );
}

/// Ports `test_anthropic_missing_messages` and the error half of every case: an undefined request
/// is a 400 naming what it refused, inside Anthropic's envelope around the OpenAI path's error
/// object — where llama-server defaults or drops: a missing `max_tokens`, one below 1 or not an
/// integer, a block of an unknown type (in a message, in the system prompt, in a tool result), a
/// tool without a name, with a schema that is not an object, or of a server type, an unknown
/// `tool_choice` or `thinking` type, an enabled `thinking` without its budget, an image source of
/// another type. A forced tool call is the chat path's refusal; a prompt past the context its
/// `exceed_context_size_error`; a stream refused before it starts gets the plain answer.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_errors() {
    let addr = start(64);
    let user = |content: Value| json!({"messages": [{"role": "user", "content": content}]});
    let block = |b: Value| turns(user(json!([b])));
    let tool = |t: Value| turns(json!({"tools": [t]}));
    let cases = [
        (
            json!({"model": "m", "max_tokens": 8}),
            "'messages' is required",
        ),
        (
            turns(json!({"messages": "hi"})),
            "'messages' must be an array",
        ),
        (turns(json!({"max_tokens": null})), "max_tokens is required"),
        (
            turns(json!({"max_tokens": 0})),
            "max_tokens must be at least 1, not 0",
        ),
        (
            turns(json!({"max_tokens": "8"})),
            "max_tokens must be an integer, not a string",
        ),
        (
            turns(json!({"max_tokens": 2.5})),
            "max_tokens must be an integer",
        ),
        (
            block(json!({"type": "document", "source": {}})),
            "content block type \"document\" is not supported",
        ),
        (block(json!({"text": "untyped"})), "a string 'type'"),
        (
            block(json!({"type": "text", "text": 7})),
            "a text block needs a string 'text'",
        ),
        (
            turns(json!({"system": [{"type": "image", "source": {}}]})),
            "a system block of type \"image\"",
        ),
        (
            block(
                json!({"type": "tool_result", "tool_use_id": "t", "content": [
                    {"type": "document"},
                ]}),
            ),
            "a tool_result block of type \"document\"",
        ),
        (
            block(json!({"type": "image", "source": {"type": "file", "file_id": "f"}})),
            "image source type \"file\"",
        ),
        (
            tool(json!({"description": "d", "input_schema": {"type": "object"}})),
            "each tool needs a non-empty string 'name'",
        ),
        (
            tool(json!({"name": "f", "input_schema": 3})),
            "tool f: 'input_schema' must be an object",
        ),
        (
            tool(json!({"type": "web_search_20250305", "name": "web_search"})),
            "tool type \"web_search_20250305\" is not supported",
        ),
        (turns(json!({"tools": {}})), "tools must be an array"),
        (
            turns(json!({"tool_choice": {"type": "sometimes"}})),
            "tool_choice must be an object whose type is",
        ),
        (
            turns(json!({"tools": [{"name": "f"}], "tool_choice": {"type": "any"}})),
            "tool_choice is not supported by this server",
        ),
        (
            turns(json!({"thinking": {"type": "enabled"}})),
            "thinking.budget_tokens is required",
        ),
        (
            turns(json!({"thinking": {"type": "enabled", "budget_tokens": "9"}})),
            "thinking.budget_tokens must be an integer",
        ),
        (
            turns(json!({"thinking": {"type": "always"}})),
            "thinking must be an object whose type is",
        ),
        (
            turns(json!({"stop_sequences": "END"})),
            "stop_sequences must be an array of strings",
        ),
    ];
    for (body, part) in cases {
        for stream in [false, true] {
            let b = with(body.clone(), json!({"stream": stream}));
            assert_envelope(
                &post(addr, "/v1/messages", &b),
                400,
                "invalid_request_error",
                part,
            );
        }
    }
    let mut no_max = turns(json!({}));
    no_max
        .as_object_mut()
        .expect("an object")
        .remove("max_tokens");
    assert_envelope(
        &post(addr, "/v1/messages", &no_max),
        400,
        "invalid_request_error",
        "max_tokens is required",
    );
    assert_envelope(
        &call(addr, "POST", "/v1/messages", Some("{not json")),
        400,
        "invalid_request_error",
        "invalid JSON body",
    );
    assert_envelope(
        &post(addr, "/v1/messages", &turns(user(json!("x".repeat(64))))),
        400,
        "exceed_context_size_error",
        "the context holds 64",
    );
}

/// An engine that fails mid-generation ends a stream that has started with an `error` event in
/// the envelope, after the text sent so far and with no `message_stop`; a request that has not
/// started gets the envelope as its answer.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_engine_error_mid_stream() {
    let addr = start_with(Box::new(MockEngine::failing_at(4096, 2)));
    let r = post(addr, "/v1/messages", &turns(json!({"stream": true})));
    let f = frames(&r);
    assert_eq!(
        types(&f),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "error"
        ],
        "{}",
        r.body
    );
    assert_eq!(joined(&f, "text_delta", "text"), "a", "{}", r.body);
    let e = &f[3];
    assert_eq!(e["error"]["code"], 500, "{e}");
    assert_eq!(e["error"]["type"], "server_error", "{e}");
    assert!(
        e["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("injected failure")),
        "{e}"
    );
    assert_envelope(
        &post(addr, "/v1/messages", &turns(json!({}))),
        503,
        "unavailable_error",
        "the engine failed",
    );
}

/// The system prompt's billing header (`x-anthropic-billing-header: …; cch=<5>;`, Claude Code's
/// first system text) renders with its stamp as `fffff`, as a string and as the first of several
/// blocks, so two requests whose stamps differ render one prompt and the second keeps all but its
/// last id in the cache (llama-server's `normalize_anthropic_billing_header`).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_billing_header_keeps_the_prompt_cache() {
    let addr = start(4096);
    let head = "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; ";
    let ask = |system: Value| {
        let v = post(addr, "/v1/messages", &turns(json!({"system": system}))).json();
        (v["usage"].clone(), last_prompt(addr))
    };
    let (_, first) = ask(json!(format!("{head}cch=a5145;You are Claude Code.")));
    let (usage, second) = ask(json!([
        {"type": "text", "text": format!("{head}cch=0b7e2;")},
        {"type": "text", "text": "You are Claude Code."},
    ]));
    assert_eq!(first, second);
    let prompt = first.as_str().expect("a prompt");
    assert!(
        prompt.contains(&format!("{head}cch=fffff;You are Claude Code.")),
        "{prompt}"
    );
    let n = usage["input_tokens"].as_u64().expect("input")
        + usage["cache_read_input_tokens"].as_u64().expect("cache");
    assert_eq!(usage["cache_read_input_tokens"], n - 1, "{usage}");
}

/// Ports `test_anthropic_vision_format_accepted` and `test_anthropic_tool_result_with_image`: an
/// `image` block — a `base64` source as a `data:` URL, a `url` source as its URL, in a message or in
/// a tool result — is answered as the OpenAI path answers the matching `image_url` part: the same
/// status, `type` and `message`, inside the envelope.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_vision_format_accepted() {
    let addr = start(4096);
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";
    let data = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}});
    let data_part =
        json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png}")}});
    let url =
        json!({"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}});
    let url_part = json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}});
    let text = json!({"type": "text", "text": "What is this?"});
    let question = json!({"role": "user", "content": "What is in this image?"});
    let call = json!({"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"file": "a.png"}},
    ]});
    let openai_call = json!({"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{\"file\":\"a.png\"}"}},
    ]});
    let cases = [
        (
            json!([{"role": "user", "content": [data.clone(), text.clone()]}]),
            json!([{"role": "user", "content": [data_part.clone(), text.clone()]}]),
        ),
        (
            json!([{"role": "user", "content": [url, text.clone()]}]),
            json!([{"role": "user", "content": [url_part, text.clone()]}]),
        ),
        (
            json!([question.clone(), call, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "text", "text": "File: a.png"}, data,
                ]},
            ]}]),
            json!([question, openai_call, {"role": "tool", "tool_call_id": "t1", "content": [
                {"type": "text", "text": "File: a.png"}, data_part,
            ]}]),
        ),
    ];
    for (anthropic, openai) in cases {
        let a = post(
            addr,
            "/v1/messages",
            &json!({"max_tokens": 10, "messages": anthropic}),
        );
        let o = post(
            addr,
            "/v1/chat/completions",
            &json!({"max_tokens": 10, "messages": openai}),
        );
        assert_eq!(a.status, o.status, "{} vs {}", a.body, o.body);
        let ae = a.json();
        assert_eq!(ae["type"], "error", "{ae}");
        assert_eq!(ae["error"], o.json()["error"], "{anthropic}");
    }
}

/// Ports `test_anthropic_temperature`, `test_anthropic_top_p`, `test_anthropic_top_k` and the
/// request half of `test_anthropic_stop_sequences`: `temperature`, `top_p`, `top_k`,
/// `stop_sequences` and `max_tokens` reach the generation as the OpenAI path's `temperature`,
/// `top_p`, `top_k`, `stop` and `max_tokens` do (the slot's settings, the seed aside).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_anthropic_sampling_fields_reach_the_generation() {
    let addr = start(4096);
    let settings = |path: &str, b: &Value| {
        let r = post(addr, path, b);
        assert_eq!(r.status, 200, "{path}: {}", r.body);
        let mut s: Map<String, Value> = get(addr, "/slots").json()[0]
            .as_object()
            .expect("a slot")
            .clone();
        for k in [
            "seed",
            "id_task",
            "task_id",
            "prompt",
            "next_token",
            "n_past",
        ] {
            s.remove(k);
        }
        s
    };
    let fields = json!({"temperature": 0.5, "top_p": 0.9, "top_k": 40, "max_tokens": 3});
    let anthropic = settings(
        "/v1/messages",
        &with(
            turns(fields.clone()),
            json!({"stop_sequences": ["\n", "END"]}),
        ),
    );
    let openai = settings(
        "/v1/chat/completions",
        &with(turns(fields), json!({"stop": ["\n", "END"]})),
    );
    assert_eq!(anthropic, openai);
    assert_eq!(anthropic["top_k"], 40, "{anthropic:?}");
    assert_eq!(anthropic["n_predict"], 3, "{anthropic:?}");
    assert_eq!(anthropic["stop"], json!(["\n", "END"]), "{anthropic:?}");
}
