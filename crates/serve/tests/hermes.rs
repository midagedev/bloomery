//! Gate: Hermes tool calls and a model-opened think span on the Qwen3 chat
//! template (`tests/fixtures/qwen3-chat-template.jinja`, the Qwen3 family's
//! — llama.cpp ships it as `models/templates/Qwen-Qwen3-0.6B.jinja`, and the
//! Qwen3-30B-A3B-Thinking-2507 GGUF's template spells the same span and call
//! markup). Its `tools` header and assistant `tool_calls` branch teach:
//!
//! ```text
//! <tool_call>
//! {"name": "get_weather", "arguments": {"location": "Seoul"}}
//! </tool_call>
//! ```
//!
//! and its thinking-on generation prompt ends at `<|im_start|>assistant\n`:
//! the model opens `<think>` itself, so the reasoning split — and its budget
//! — starts at the model's tag, as llama-server's parser takes an optional
//! leading span and its budget sampler starts counting at the start tag.

mod common;

use common::{V41_TEMPLATE, post, start_templated};
use serde_json::{Value, json};
use serve::dsml::{ChatParser, MarkupError, Message, ToolCall, ToolFormat, Tools};
use serve::hermes::HermesError;
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const TEMPLATE: &str = include_str!("fixtures/qwen3-chat-template.jinja");
const QWEN38: &str = include_str!("fixtures/qwen38-chat-template.jinja");
const GLM5: &str = include_str!("fixtures/glm5-chat-template.jinja");

/// The template's generation prompt after one user turn, thinking on: the
/// span is the model's to open.
const ON: &str = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n";
/// A prompt the template closed (thinking off writes the empty span).
const CLOSED: &str =
    "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

const ONE_CALL: &str = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Seoul\"}}\n</tool_call>";
const TWO_CALLS: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Seoul\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
const STRING_ARGS: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"Seoul\\\", \\\"days\\\": 3}\"}\n</tool_call>";
const THINK_THEN_CALL: &str = "<think>Need the weather.</think><tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
const CALL_THEN_TEXT: &str =
    "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call> done.";
const ANSWER: &str = "Hi.";
const NEAR_TAG: &str = "a <tool_ca b <tool_call";

fn call(index: usize, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        index,
        name: name.to_owned(),
        arguments: arguments.to_owned(),
    }
}

fn msg(reasoning: &str, content: &str, calls: Vec<ToolCall>) -> Message {
    Message {
        reasoning: reasoning.to_owned(),
        content: content.to_owned(),
        calls,
    }
}

/// `(name, prompt, format, raw output, expected message)`.
fn fixtures() -> Vec<(
    &'static str,
    &'static str,
    ReasoningFormat,
    &'static str,
    Message,
)> {
    vec![
        (
            "text then one call",
            ON,
            ReasoningFormat::Deepseek,
            ONE_CALL,
            msg(
                "",
                "Let me check.\n",
                vec![call(0, "get_weather", r#"{"location":"Seoul"}"#)],
            ),
        ),
        (
            "two calls",
            ON,
            ReasoningFormat::Deepseek,
            TWO_CALLS,
            msg(
                "",
                "",
                vec![
                    call(0, "get_weather", r#"{"location":"Seoul"}"#),
                    call(1, "get_time", "{}"),
                ],
            ),
        ),
        (
            "arguments as the template's string spelling",
            ON,
            ReasoningFormat::Deepseek,
            STRING_ARGS,
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            ),
        ),
        (
            "a model-opened span before the call",
            ON,
            ReasoningFormat::Deepseek,
            THINK_THEN_CALL,
            msg("Need the weather.", "", vec![call(0, "get_time", "{}")]),
        ),
        (
            "text after a call stays content",
            ON,
            ReasoningFormat::Deepseek,
            CALL_THEN_TEXT,
            msg("", " done.", vec![call(0, "get_time", "{}")]),
        ),
        (
            "no call",
            ON,
            ReasoningFormat::Deepseek,
            ANSWER,
            msg("", "Hi.", vec![]),
        ),
        (
            "text that is not a call opening, and one cut short at the end",
            ON,
            ReasoningFormat::Deepseek,
            NEAR_TAG,
            msg("", "a <tool_ca b <tool_call", vec![]),
        ),
        (
            "a span the prompt closed",
            CLOSED,
            ReasoningFormat::Deepseek,
            ANSWER,
            msg("", "Hi.", vec![]),
        ),
    ]
}

/// `(name, raw output, the error)`, each after the prompt [`ON`].
fn malformed() -> Vec<(&'static str, &'static str, HermesError)> {
    vec![
        (
            "unterminated",
            "<tool_call>\n{\"name\": \"get_time\"",
            HermesError::Unterminated {
                index: 0,
                text: "<tool_call>\n{\"name\": \"get_time\"".to_owned(),
            },
        ),
        (
            "a body that is not JSON",
            "<tool_call>not json</tool_call>",
            HermesError::NotJson {
                index: 0,
                found: "not json".to_owned(),
            },
        ),
        (
            "a body of JSON that is not an object",
            "<tool_call>[1, 2]</tool_call>",
            HermesError::NotJson {
                index: 0,
                found: "[1, 2]".to_owned(),
            },
        ),
        (
            "no name",
            "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
            HermesError::NoName {
                index: 0,
                found: "{\"arguments\": {}}".to_owned(),
            },
        ),
        (
            "no arguments",
            "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
            HermesError::NoArguments {
                index: 0,
                name: "f".to_owned(),
                found: "absent".to_owned(),
            },
        ),
        (
            "arguments that are neither object nor string",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": 3}\n</tool_call>",
            HermesError::NoArguments {
                index: 0,
                name: "f".to_owned(),
                found: "3".to_owned(),
            },
        ),
        (
            "a string argument holding no object",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": \"x\"}\n</tool_call>",
            HermesError::NoArguments {
                index: 0,
                name: "f".to_owned(),
                found: "x".to_owned(),
            },
        ),
        (
            "an unknown key",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {}, \"id\": 5}\n</tool_call>",
            HermesError::UnknownKey {
                index: 0,
                name: "f".to_owned(),
                key: "id".to_owned(),
            },
        ),
        (
            "the second call malformed",
            "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>\n<tool_call>x</tool_call>",
            HermesError::NotJson {
                index: 1,
                found: "x".to_owned(),
            },
        ),
    ]
}

/// Feeds `pieces` one by one; the deltas' sum, checked against the message.
fn streamed(
    prompt: &str,
    format: ReasoningFormat,
    pieces: &[&str],
) -> Result<Message, MarkupError> {
    let mut p = ChatParser::with_tools(prompt, format, Some(Tools::Hermes));
    let mut total = Message::default();
    let mut add = |d: Message| {
        total.reasoning.push_str(&d.reasoning);
        total.content.push_str(&d.content);
        total.calls.extend(d.calls);
    };
    for piece in pieces {
        add(p.try_push(piece)?);
    }
    add(p.try_finish()?);
    assert_eq!(&total, p.message(), "deltas sum to the message");
    Ok(total)
}

/// The whole text at once, the text a character at a time, split in two at
/// every character, and split in three at every pair of tag boundaries.
fn every_chunking(raw: &str) -> Vec<Vec<&str>> {
    let mut out = vec![vec![raw]];
    out.push(
        raw.char_indices()
            .map(|(i, c)| &raw[i..i + c.len_utf8()])
            .collect(),
    );
    for (at, _) in raw.char_indices().skip(1) {
        out.push(vec![&raw[..at], &raw[at..]]);
    }
    let tags: Vec<usize> = raw
        .char_indices()
        .filter(|&(_, c)| c == '<' || c == '>')
        .map(|(i, c)| if c == '>' { i + 1 } else { i })
        .filter(|&i| i > 0 && i < raw.len())
        .collect();
    for (x, &a) in tags.iter().enumerate() {
        for &b in &tags[x + 1..] {
            if a < b {
                out.push(vec![&raw[..a], &raw[a..b], &raw[b..]]);
            }
        }
    }
    out
}

/// The fixtures parse to their messages, their arguments JSON objects.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_fixtures_parse_to_expected_messages() {
    let mut wrong = Vec::new();
    for (name, prompt, format, raw, want) in fixtures() {
        match streamed(prompt, format, &[raw]) {
            Ok(got) if got == want => {
                for c in &got.calls {
                    let v: Value = serde_json::from_str(&c.arguments).expect("arguments are JSON");
                    assert!(v.is_object(), "{name}: arguments are an object");
                }
            }
            got => wrong.push(format!("{name}:\n  got  {got:?}\n  want {want:?}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} fixture(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Markup that does not parse is a named error, never content and never a
/// guessed call.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_malformed_calls_are_named_errors() {
    let mut wrong = Vec::new();
    for (name, raw, want) in malformed() {
        match streamed(ON, ReasoningFormat::Deepseek, &[raw]) {
            Err(e) if e == MarkupError::Hermes(want.clone()) => {}
            got => wrong.push(format!("{name}:\n  got  {got:?}\n  want Err({want:?})")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A message, or an error, never depends on how the text was cut: whole, a
/// character at a time, in two at every character, in three at every pair of
/// tag boundaries (a call split across three stream pieces at its tags).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_parse_does_not_depend_on_chunking() {
    let cases = fixtures()
        .into_iter()
        .map(|(name, prompt, format, raw, want)| (name, prompt, format, raw, Ok(want)))
        .chain(malformed().into_iter().map(|(name, raw, e)| {
            (
                name,
                ON,
                ReasoningFormat::Deepseek,
                raw,
                Err(MarkupError::from(e)),
            )
        }));
    for (name, prompt, format, raw, want) in cases {
        for pieces in every_chunking(raw) {
            assert_eq!(
                streamed(prompt, format, &pieces),
                want,
                "{name}: {pieces:?}"
            );
        }
    }
}

/// The server reads the markup from its template: Hermes' for Qwen3's, whose
/// source spells `<tool_call>` with neither GLM's nor the DSML token — the
/// rendered call names it — and GLM's for the GLM template, which spells
/// `<tool_call>` too: its tags come first, and its rendered call body is not
/// JSON. V4.1 stays DSML (the templates the source spells for another family
/// — Qwen's XML among them — have their own gates).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_detection_follows_the_template() {
    let parsed = |source: &str| ChatTemplate::parse(source).expect("template parses");
    let qwen3 = parsed(TEMPLATE);
    assert_eq!(ToolFormat::of_chat_template(&qwen3), ToolFormat::Hermes);
    assert_eq!(
        ToolFormat::of_chat_template(&parsed(GLM5)),
        ToolFormat::GlmXml
    );
    assert_eq!(
        ToolFormat::of_chat_template(&parsed(V41_TEMPLATE)),
        ToolFormat::Dsml
    );
    // The source spellings alone type none of the `<tool_call>` templates
    // left to the probe.
    assert_eq!(ToolFormat::of_template(TEMPLATE), ToolFormat::Unparsed);
}

// ---------------------------------------------------------------- the server

fn start(script: &str) -> std::net::SocketAddr {
    start_templated(Box::new(ScriptedEngine::new(4096, script)), TEMPLATE)
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object",
            "properties": {}}}},
    ])
}

fn tool_body(stream: bool) -> Value {
    json!({
        "model": "m",
        "messages": [{"role": "user", "content": "weather and time?"}],
        "temperature": 0,
        "tools": tools(),
        "stream": stream,
    })
}

/// A streamed chat body folded back into `(finish_reason, message)`, call ids
/// left out; `Err` with its message when the stream ended on an error event.
fn fold_stream(body: &str) -> Result<(Value, Value), String> {
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut finish = Value::Null;
    for e in body.split("\n\n").filter_map(|e| e.strip_prefix("data: ")) {
        if e == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(e).expect("chunk JSON");
        if let Some(m) = v["error"]["message"].as_str() {
            assert!(!body.contains("[DONE]"), "an error ends the stream: {body}");
            return Err(m.to_owned());
        }
        let Some(c) = v["choices"].get(0) else {
            continue;
        };
        if !c["finish_reason"].is_null() {
            finish = c["finish_reason"].clone();
        }
        let d = &c["delta"];
        if let Some(s) = d["reasoning_content"].as_str() {
            reasoning.push_str(s);
        }
        if let Some(s) = d["content"].as_str() {
            assert!(
                !s.contains("<tool_call>"),
                "markup reached delta.content: {s}"
            );
            content.push_str(s);
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            calls.push(tc["function"].clone());
        }
    }
    let mut m = json!({"content": content, "reasoning_content": reasoning});
    m["calls"] = json!(calls);
    Ok((finish, m))
}

/// One and two calls, streamed and not: `tool_calls` in OpenAI's shape with
/// `finish_reason` `tool_calls`, the text before them the content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_hermes_chat_tool_calls_stream_and_non_stream() {
    for (script, want) in [
        (
            ONE_CALL,
            vec![json!({"name": "get_weather", "arguments": r#"{"location":"Seoul"}"#})],
        ),
        (
            TWO_CALLS,
            vec![
                json!({"name": "get_weather", "arguments": r#"{"location":"Seoul"}"#}),
                json!({"name": "get_time", "arguments": "{}"}),
            ],
        ),
    ] {
        let addr = start(script);
        let r = post(addr, "/v1/chat/completions", &tool_body(false));
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let c = &v["choices"][0];
        assert_eq!(c["finish_reason"], "tool_calls", "{v}");
        let m = &c["message"];
        assert!(m.get("reasoning_content").is_none(), "{v}");
        assert_eq!(m["content"], want_content(script), "{v}");
        let calls: Vec<Value> = m["tool_calls"]
            .as_array()
            .expect("tool_calls")
            .iter()
            .map(|tc| tc["function"].clone())
            .collect();
        assert_eq!(calls, want, "{v}");
        for tc in m["tool_calls"].as_array().expect("tool_calls") {
            assert_eq!(tc["type"], "function", "{v}");
            assert!(
                tc["id"].as_str().is_some_and(|i| i.starts_with("call_")),
                "{v}"
            );
        }
        let r = post(addr, "/v1/chat/completions", &tool_body(true));
        assert_eq!(r.status, 200, "{}", r.body);
        let (finish, folded) = fold_stream(&r.body).expect("no error event");
        assert_eq!(finish, "tool_calls");
        assert_eq!(
            folded["content"],
            want_content(script),
            "streamed == non-streamed"
        );
        assert_eq!(folded["calls"], json!(want), "streamed == non-streamed");
    }
}

/// The content of a fixture's script: its text before the first call.
fn want_content(script: &str) -> Value {
    let at = script.find("<tool_call>").expect("a call");
    json!(script[..at])
}

/// `/v1/messages` answers the same generation with a `tool_use` block and
/// `stop_reason` `tool_use`, the arguments an object.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_hermes_messages_tool_use() {
    let addr = start(ONE_CALL);
    let b = json!({
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "weather?"}],
        "temperature": 0,
        "tools": [{"name": "get_weather", "description": "",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}}],
    });
    let r = post(addr, "/v1/messages", &b);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(v["stop_reason"], "tool_use", "{v}");
    assert_eq!(v["role"], "assistant", "{v}");
    let blocks = v["content"].as_array().expect("content blocks");
    let last = blocks.last().expect("a block");
    assert_eq!(last["type"], "tool_use", "{v}");
    assert_eq!(last["name"], "get_weather", "{v}");
    assert_eq!(last["input"], json!({"location": "Seoul"}), "{v}");
    assert!(
        last["id"].as_str().is_some_and(|i| i.starts_with("toolu_")),
        "{v}"
    );
    let text = blocks
        .iter()
        .find(|b| b["type"] == "text")
        .expect("the text before the call");
    assert_eq!(text["text"], "Let me check.\n", "{v}");
}

/// Markup that does not parse is the request's named error: a 500 without a
/// stream, an error event that ends a stream (no `[DONE]`), for a malformed
/// call and for one the generation leaves open.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_hermes_malformed_call_is_an_api_error() {
    for (script, want) in [
        (
            "<tool_call>\n{\"name\": \"f\", \"arguments\": 3}\n</tool_call>",
            "tool-call markup: tool call 0 (f) carries no arguments object: \"3\"",
        ),
        (
            "Calling.<tool_call>\n{\"name\": \"f\"",
            "tool-call markup: tool call 0 is not closed",
        ),
    ] {
        let addr = start(script);
        let r = post(addr, "/v1/chat/completions", &tool_body(false));
        assert_eq!(r.status, 500, "{}", r.body);
        let m = r.json()["error"]["message"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        assert!(m.starts_with(want), "{m}");
        let r = post(addr, "/v1/chat/completions", &tool_body(true));
        let e = fold_stream(&r.body).expect_err("an error event");
        assert!(e.starts_with(want), "{e}");
    }
}

// ---------------------------------------------------------------- the span

/// A chat body on the Qwen3 template with `extra` beside it.
fn qwen_chat(extra: Value) -> Value {
    let mut b = json!({
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0,
        "max_tokens": 32,
    });
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e.clone());
    }
    b
}

/// A span the model opens itself splits as a prompt-opened one does:
/// `reasoning_content` and `content` on the OpenAI path, a `thinking` block
/// before the `text` block on `/v1/messages`, streamed and not.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_model_opened_span_splits_like_a_prompt_opened_one() {
    const SCRIPT: &str = "<think>Need the weather.</think>Hello!";
    let addr = start(SCRIPT);
    let r = post(addr, "/v1/chat/completions", &qwen_chat(json!({})));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "Need the weather.", "{v}");
    assert_eq!(m["content"], "Hello!", "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    let r = post(
        addr,
        "/v1/chat/completions",
        &qwen_chat(json!({"stream": true})),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "stop");
    assert_eq!(folded["reasoning_content"], "Need the weather.");
    assert_eq!(folded["content"], "Hello!");

    let mut b = json!({
        "model": "m", "max_tokens": 32, "temperature": 0,
        "messages": [{"role": "user", "content": "q"}],
    });
    let addr = start(SCRIPT);
    let r = post(addr, "/v1/messages", &b);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(
        v["content"],
        json!([
            {"type": "thinking", "thinking": "Need the weather.", "signature": ""},
            {"type": "text", "text": "Hello!"},
        ]),
        "{v}"
    );
    assert_eq!(v["stop_reason"], "end_turn", "{v}");
    b["stream"] = json!(true);
    let r = post(addr, "/v1/messages", &b);
    assert_eq!(r.status, 200, "{}", r.body);
    // The frames' event names, a delta frame's payload naming its kind.
    fn kind(v: &Value) -> &str {
        v["delta"]["type"]
            .as_str()
            .or_else(|| v["type"].as_str())
            .expect("an event type")
    }
    let frames = message_frames(&r);
    let kinds: Vec<&str> = frames.iter().map(kind).collect();
    assert_eq!(kinds.first(), Some(&"message_start"), "{}", r.body);
    assert_eq!(kinds.last(), Some(&"message_stop"), "{}", r.body);
    // The thinking block opens the message and closes before the text block,
    // its empty signature delta its last word.
    let thinking = kinds
        .iter()
        .position(|k| *k == "thinking_delta")
        .expect("a thinking delta");
    let signature = kinds
        .iter()
        .position(|k| *k == "signature_delta")
        .expect("a signature delta");
    let text = kinds
        .iter()
        .position(|k| *k == "text_delta")
        .expect("a text delta");
    assert!(
        kinds[..thinking].contains(&"content_block_start"),
        "{}",
        r.body
    );
    assert!(
        thinking < signature && signature < text,
        "thinking, its signature, then text: {}",
        r.body
    );
}

/// The data objects of a `/v1/messages` stream, in order: every frame is an
/// `event:` line and a `data:` line whose JSON's `type` is the event's name.
fn message_frames(r: &common::Reply) -> Vec<Value> {
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

/// A generation with no span passes through: the model that answers without
/// its `<think>` keeps the whole text as content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_model_that_opens_no_span_answers_in_content() {
    let addr = start("Plain answer.");
    let r = post(addr, "/v1/chat/completions", &qwen_chat(json!({})));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let m = &v["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "{v}");
    assert_eq!(m["content"], "Plain answer.", "{v}");
}

/// The budget caps a model-opened span: the opening tag spends nothing, the
/// next four ids are the reasoning, the close id is forced as the sixth taken
/// token (the answer it displaces never appears), and the split consumes the
/// tags — `</think>` shows in neither field.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_the_budget_caps_a_model_opened_span() {
    let addr = start("<think>abcdefghijkl");
    let r = post(
        addr,
        "/v1/chat/completions",
        &qwen_chat(json!({"reasoning_budget": 4})),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let m = &v["choices"][0]["message"];
    let (reasoning, content) = (
        m["reasoning_content"].as_str().unwrap_or(""),
        m["content"].as_str().unwrap_or(""),
    );
    assert_eq!(reasoning, "abcd", "{v}");
    assert_eq!(content, "fghijkl", "{v}");
    assert!(
        !reasoning.contains("</think>") && !content.contains("</think>"),
        "{v}"
    );
    // The script is 13 ids and the eos: 14 taken.
    assert_eq!(v["usage"]["completion_tokens"], json!(14), "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], json!("stop"), "{v}");
}

/// A prompt the template already closed (thinking off) silently ignores the
/// budget, as llama-server ignores the flag when thinking is off by other
/// means: no close id forced, the whole script the content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_the_budget_is_ignored_on_a_closed_span_prompt() {
    let addr = start("<think>abcdefghijkl");
    let r = post(
        addr,
        "/v1/chat/completions",
        &qwen_chat(json!({
            "reasoning_budget": 0,
            "chat_template_kwargs": {"enable_thinking": false},
        })),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let m = &v["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "{v}");
    assert_eq!(m["content"], json!("<think>abcdefghijkl"), "{v}");
    assert_eq!(v["usage"]["completion_tokens"], json!(14), "{v}");
}

/// A prompt ending `<think>\n` (Qwen3.8's thinking-on prompt) starts the model
/// inside the span, its newline with it: the script's first text is already
/// reasoning, no model-opened tag needed.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_prompt_ending_think_newline_opens_the_span() {
    let addr = start_templated(
        Box::new(ScriptedEngine::new(
            4096,
            "The user greets me.</think>Hello!",
        )),
        QWEN38,
    );
    let r = post(
        addr,
        "/v1/chat/completions",
        &json!({
            "messages": [{"role": "user", "content": "q"}],
            "temperature": 0,
            "max_tokens": 32,
        }),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "The user greets me.", "{v}");
    assert_eq!(m["content"], "Hello!", "{v}");
}
