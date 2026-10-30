//! Gate: Mistral tool calls on the Mistral Large 4 chat template
//! (`tests/fixtures/mistral-large4-chat-template.jinja`, the checkpoint's own
//! `chat_template.jinja`). Its assistant `tool_calls` branch teaches:
//!
//! ```text
//! [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}
//! ```
//!
//! one markup per call, no call id, nothing between calls, the arguments one
//! JSON value with no close tag: their own brackets, quotes and words end
//! them, so a `[TOOL_CALLS]` inside a string value is text. The model opens
//! no think span; the reasoning split is off.

mod common;

use common::{post, start_templated};
use serde_json::{Value, json};
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, ToolCall, ToolFormat, Tools};
use serve::mistral::MistralError;
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const TEMPLATE: &str = include_str!("fixtures/mistral-large4-chat-template.jinja");

/// The prompt after one user turn, as the template renders it.
const ON: &str = "<s>[INST]q[/INST]";

const ONE_CALL: &str =
    "Let me check. [TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", \"unit\": \"C\"}";
const TWO_CALLS: &str =
    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}[TOOL_CALLS]get_time[ARGS]{}";
const NESTED: &str = "[TOOL_CALLS]search[ARGS]{\"q\": \"x\", \"tags\": [\"a\", 2, null], \"where\": {\"city\": \"Paris\", \"deep\": true, \"hits\": 1.5}}";
const PRETTY: &str = "[TOOL_CALLS]get_time[ARGS]  {\n  \"tz\": \"UTC\"\n}";
const STRING_VALUE: &str = "[TOOL_CALLS]echo[ARGS]\"[TOOL_CALLS]x[ARGS]1\"";
const NUMBER_VALUE: &str = "[TOOL_CALLS]set_level[ARGS]7 done.";
const TEXT_AROUND: &str =
    "Before. [TOOL_CALLS]get_time[ARGS]{} Between. [TOOL_CALLS]get_time[ARGS]{} After.";
const ANSWER: &str = "Hi.";
const NEAR_TAG: &str = "a [TOOL_CAL b [TOOL_CALL";

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

/// `(name, raw output, expected message)`.
fn fixtures() -> Vec<(&'static str, &'static str, Message)> {
    vec![
        (
            "text then one call",
            ONE_CALL,
            msg(
                "",
                "Let me check. ",
                vec![call(0, "get_weather", r#"{"city":"Paris","unit":"C"}"#)],
            ),
        ),
        (
            "two calls",
            TWO_CALLS,
            msg(
                "",
                "",
                vec![
                    call(0, "get_weather", r#"{"city":"Paris"}"#),
                    call(1, "get_time", "{}"),
                ],
            ),
        ),
        (
            "nested JSON arguments",
            NESTED,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "search",
                    r#"{"q":"x","tags":["a",2,null],"where":{"city":"Paris","deep":true,"hits":1.5}}"#,
                )],
            ),
        ),
        (
            "whitespace inside the arguments",
            PRETTY,
            msg("", "", vec![call(0, "get_time", r#"{"tz":"UTC"}"#)]),
        ),
        (
            "a string value holding the markup",
            STRING_VALUE,
            msg("", "", vec![call(0, "echo", r#""[TOOL_CALLS]x[ARGS]1""#)]),
        ),
        (
            "a number value the trailing text ends",
            NUMBER_VALUE,
            msg("", " done.", vec![call(0, "set_level", "7")]),
        ),
        (
            "a number value the stream ends",
            "[TOOL_CALLS]set_level[ARGS]7",
            msg("", "", vec![call(0, "set_level", "7")]),
        ),
        (
            "a word value the stream ends",
            "[TOOL_CALLS]reset[ARGS]true",
            msg("", "", vec![call(0, "reset", "true")]),
        ),
        (
            "text before, between and after calls",
            TEXT_AROUND,
            msg(
                "",
                "Before.  Between.  After.",
                vec![call(0, "get_time", "{}"), call(1, "get_time", "{}")],
            ),
        ),
        ("no call", ANSWER, msg("", "Hi.", vec![])),
        (
            "text that is not a call opening",
            NEAR_TAG,
            msg("", "a [TOOL_CAL b [TOOL_CALL", vec![]),
        ),
    ]
}

/// `(name, raw output, the error)`, each after the prompt [`ON`].
fn malformed() -> Vec<(&'static str, &'static str, MistralError)> {
    vec![
        (
            "a name with no [ARGS]",
            "[TOOL_CALLS]get_weather",
            MistralError::Unterminated {
                index: 0,
                text: "[TOOL_CALLS]get_weather".to_owned(),
            },
        ),
        (
            "a name that holds a [",
            "[TOOL_CALLS]get[weather[ARGS]{}",
            MistralError::NoName {
                index: 0,
                found: "get[".to_owned(),
            },
        ),
        (
            "a name that holds a newline",
            "[TOOL_CALLS]get\nweather[ARGS]{}",
            MistralError::NoName {
                index: 0,
                found: "get\n".to_owned(),
            },
        ),
        (
            "an empty name",
            "[TOOL_CALLS][ARGS]{}",
            MistralError::NoName {
                index: 0,
                found: "".to_owned(),
            },
        ),
        (
            "arguments that begin no JSON value",
            "[TOOL_CALLS]get_time[ARGS]now",
            MistralError::NotJson {
                index: 0,
                name: "get_time".to_owned(),
                found: "no".to_owned(),
            },
        ),
        (
            "arguments whose brackets close no JSON",
            "[TOOL_CALLS]get_time[ARGS]{tz}",
            MistralError::NotJson {
                index: 0,
                name: "get_time".to_owned(),
                found: "{tz}".to_owned(),
            },
        ),
        (
            "a number left short by the stream's end",
            "[TOOL_CALLS]set_level[ARGS]1.",
            MistralError::Unterminated {
                index: 0,
                text: "[TOOL_CALLS]set_level[ARGS]1.".to_owned(),
            },
        ),
        (
            "arguments still growing at the end",
            "[TOOL_CALLS]get_time[ARGS]{\"tz\": \"UTC\"",
            MistralError::Unterminated {
                index: 0,
                text: "[TOOL_CALLS]get_time[ARGS]{\"tz\": \"UTC\"".to_owned(),
            },
        ),
        (
            "the second call opens at EOF",
            "[TOOL_CALLS]get_time[ARGS]{}[TOOL_CALLS]",
            MistralError::Unterminated {
                index: 1,
                text: "[TOOL_CALLS]".to_owned(),
            },
        ),
        (
            "the second call malformed",
            "[TOOL_CALLS]get_time[ARGS]{}[TOOL_CALLS]f[ARGS](",
            MistralError::NotJson {
                index: 1,
                name: "f".to_owned(),
                found: "(".to_owned(),
            },
        ),
        (
            "a lone markup the stream ends",
            "answer[TOOL_CALLS]",
            MistralError::Unterminated {
                index: 0,
                text: "[TOOL_CALLS]".to_owned(),
            },
        ),
    ]
}

/// Feeds `pieces` one by one; the deltas' sum, checked against the message.
fn streamed(prompt: &str, pieces: &[&str]) -> Result<Message, MarkupError> {
    let mut p = ChatParser::with_tools(prompt, ReasoningFormat::None, Some(Tools::Mistral));
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

/// The fixtures parse to their messages, their arguments JSON.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_fixtures_parse_to_expected_messages() {
    let mut wrong = Vec::new();
    for (name, raw, want) in fixtures() {
        match streamed(ON, &[raw]) {
            Ok(got) if got == want => {
                for c in &got.calls {
                    let _: Value = serde_json::from_str(&c.arguments).expect("arguments are JSON");
                    assert!(!c.name.is_empty(), "{name}: a name");
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
        match streamed(ON, &[raw]) {
            Err(e) if e == MarkupError::Mistral(want.clone()) => {}
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
/// markup boundaries (a call split across three stream pieces at its tags).
/// A number the stream ends holds its end open — `1` may still grow — so the
/// whole parse closes it at `finish` and the split parses do too.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_parse_does_not_depend_on_chunking() {
    let cases = fixtures()
        .into_iter()
        .map(|(name, raw, want)| (name, raw, Ok(want)))
        .chain(
            malformed()
                .into_iter()
                .map(|(name, raw, e)| (name, raw, Err(MarkupError::from(e)))),
        );
    for (name, raw, want) in cases {
        for pieces in common::every_chunking(raw, 1) {
            assert_eq!(streamed(ON, &pieces), want, "{name}: {pieces:?}");
        }
    }
}

/// The call a scan has opened is reported as it is read: its name once
/// `[ARGS]` is whole, the arguments' compact JSON growing to the close.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_open_calls_are_reported_as_they_are_read() {
    let mut p =
        ChatParser::with_tools(ON, ReasoningFormat::None, Some(Tools::Mistral)).streaming_calls();
    let mut pieces = Vec::new();
    p.try_push("Let me check. [TOOL_CALLS]get_weather[ARGS]{\"city\": \"Par")
        .unwrap();
    pieces.extend(p.take_pieces());
    p.try_push("is\", \"unit\": \"C\"}").unwrap();
    pieces.extend(p.take_pieces());
    let _ = p.try_finish().unwrap();
    pieces.extend(p.take_pieces());
    assert_eq!(
        pieces,
        vec![
            CallPiece::Open {
                index: 0,
                name: "get_weather".to_owned()
            },
            CallPiece::Args {
                index: 0,
                text: r#"{"city":"Par"#.to_owned()
            },
            CallPiece::Args {
                index: 0,
                text: r#"is","unit":"C"}"#.to_owned()
            },
        ]
    );
}

/// The server reads the markup from its template: Mistral's for Mistral
/// Large 4's, whose source spells `[TOOL_CALLS]` and `[ARGS]`; the models
/// crate's table names it too.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_detection_follows_the_template() {
    let parsed = ChatTemplate::parse(TEMPLATE).expect("template parses");
    assert_eq!(ToolFormat::of_chat_template(&parsed), ToolFormat::Mistral);
    assert_eq!(ToolFormat::of_template(TEMPLATE), ToolFormat::Mistral);
    assert_eq!(
        models::ToolFormat::of_template(TEMPLATE),
        Some(models::ToolFormat::Mistral)
    );
}

// ---------------------------------------------------------------- the server

fn start(script: &str) -> std::net::SocketAddr {
    start_templated(Box::new(ScriptedEngine::new(4096, script)), TEMPLATE)
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}}}}},
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
        if let Some(s) = d["content"].as_str() {
            assert!(
                !s.contains("[TOOL_CALLS]"),
                "markup reached delta.content: {s}"
            );
            content.push_str(s);
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            common::fold_call_delta(&mut calls, tc);
        }
    }
    let mut m = json!({"content": content});
    m["calls"] = json!(
        calls
            .iter()
            .map(|c| c["function"].clone())
            .collect::<Vec<_>>()
    );
    Ok((finish, m))
}

/// One and two calls, streamed and not: `tool_calls` in OpenAI's shape with
/// `finish_reason` `tool_calls`, the text before them the content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_mistral_chat_tool_calls_stream_and_non_stream() {
    for (script, want) in [
        (
            ONE_CALL,
            vec![json!({"name": "get_weather", "arguments": r#"{"city":"Paris","unit":"C"}"#})],
        ),
        (
            TWO_CALLS,
            vec![
                json!({"name": "get_weather", "arguments": r#"{"city":"Paris"}"#}),
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
        assert_eq!(
            m["content"],
            json!(script.split("[TOOL_CALLS]").next().unwrap()),
            "{v}"
        );
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
            json!(script.split("[TOOL_CALLS]").next().unwrap()),
            "streamed == non-streamed"
        );
        assert_eq!(folded["calls"], json!(want), "streamed == non-streamed");
    }
}

/// Markup that does not parse is the request's named error: a 500 without a
/// stream, an error event that ends a stream (no `[DONE]`).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_mistral_malformed_call_is_an_api_error() {
    for (script, want) in [
        (
            "[TOOL_CALLS]get_time[ARGS]now",
            "tool-call markup: tool call 0 (get_time) carries no JSON value after [ARGS]: \"no\"",
        ),
        (
            "Calling.[TOOL_CALLS]get_time",
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
