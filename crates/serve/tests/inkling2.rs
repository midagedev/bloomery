//! Gate: Inkling 2 on the server: its two chat templates render as the reference renders them
//! (`tools/ref/inkling2-chat-render.sh`, llama.cpp PR #32695's jinja engine on the Mini's and the
//! Small's `tokenizer.chat_template`), the reply's blocks parse as `common/parsers/inkling.cpp`
//! reads them, and the server splits a scripted reply into `reasoning_content`, `content` and
//! `tool_calls`. A turn is a list of blocks, each `<|message_model|>` … `<|end_message|>`: a
//! thinking block (`<|content_thinking|>`), a text block (`<|content_text|>`) and, last, one block
//! a call (`NAME<|content_invoke_tool_json|>{"name":NAME,"args":{…}}`).

mod common;

use common::{msg, post, start_templated, tool_call as call};
use serde_json::{Value, json};
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, ToolFormat, Tools};
use serve::inkling::InklingError;
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const MINI: &str = include_str!("fixtures/inkling2-mini-chat-template.jinja");
const SMALL: &str = include_str!("fixtures/inkling2-small-chat-template.jinja");
const RENDERS: &str = include_str!("fixtures/inkling2-renders.json");

const E: &str = "<|end_message|>";
const H: &str = "<|message_model|>";
const CALL: &str = "special_function<|content_invoke_tool_json|>{\"name\":\"special_function\",\"args\":{\"arg1\":1}}<|end_message|>";

/// `(name, reasoning format, raw output, expected message)`; the first nine are the reference's own
/// cases (`tests/test-chat.cpp`, "Inkling 1 and 2").
fn fixtures() -> Vec<(&'static str, &'static str, ReasoningFormat, String, Message)> {
    use ReasoningFormat::{Deepseek, None as Keep};
    let hello = "Hello, world!\nWhat's up?";
    let think_then_text =
        format!("{H}<|content_thinking|>I'm thinking{E}{H}<|content_text|>{hello}{E}");
    vec![
        (
            "content only",
            "",
            Keep,
            format!("{H}<|content_text|>{hello}{E}"),
            msg("", hello, vec![]),
        ),
        (
            "a thinking block, then a content block",
            "",
            Deepseek,
            think_then_text.clone(),
            msg("I'm thinking", hello, vec![]),
        ),
        (
            "without reasoning extraction the thinking block stays in the content",
            "",
            Keep,
            think_then_text,
            msg(
                "",
                &format!("<|content_thinking|>I'm thinking{E}{hello}"),
                vec![],
            ),
        ),
        (
            "cut inside thinking",
            "",
            Deepseek,
            format!("{H}<|content_thinking|>I was still thinking"),
            msg("I was still thinking", "", vec![]),
        ),
        (
            "cut inside content",
            "",
            Deepseek,
            format!("{H}<|content_thinking|>I'm thinking{E}{H}<|content_text|>Hello, wor"),
            msg("I'm thinking", "Hello, wor", vec![]),
        ),
        (
            "a single call",
            "",
            Keep,
            format!("{H}{CALL}"),
            msg("", "", vec![call(0, "special_function", r#"{"arg1":1}"#)]),
        ),
        (
            "thinking, content, then a call",
            "",
            Deepseek,
            format!(
                "{H}<|content_thinking|>I should call it{E}{H}<|content_text|>On it.{E}{H}{CALL}"
            ),
            msg(
                "I should call it",
                "On it.",
                vec![call(0, "special_function", r#"{"arg1":1}"#)],
            ),
        ),
        (
            "parallel calls, one block each",
            "",
            Keep,
            format!(
                "{H}{CALL}{H}special_function_with_opt<|content_invoke_tool_json|>\
                 {{\"name\":\"special_function_with_opt\",\"args\":{{\"arg1\":1,\"arg2\":2}}}}{E}"
            ),
            msg(
                "",
                "",
                vec![
                    call(0, "special_function", r#"{"arg1":1}"#),
                    call(1, "special_function_with_opt", r#"{"arg1":1,"arg2":2}"#),
                ],
            ),
        ),
        (
            "the block closer inside a string argument does not end the block",
            "",
            Keep,
            format!(
                "{H}python<|content_invoke_tool_json|>{{\"name\":\"python\",\"args\":\
                 {{\"code\":\"print('<|end_message|>')\"}}}}{E}"
            ),
            msg(
                "",
                "",
                vec![call(0, "python", r#"{"code":"print('<|end_message|>')"}"#)],
            ),
        ),
        (
            "text blocks of one turn are one content",
            "",
            Keep,
            format!("{H}<|content_text|>One. {E}{H}<|content_text|>Two.{E}"),
            msg("", "One. Two.", vec![]),
        ),
        (
            "nested and non-ASCII arguments, whitespace between the tokens of the object",
            "",
            Keep,
            format!(
                "{H}save<|content_invoke_tool_json|>{{\"name\":\"save\",\"args\": {{\"note\": \"é 한\", \"tags\": [1, null], \"where\": {{\"a\": true}}}}}}{E}"
            ),
            msg(
                "",
                "",
                vec![call(
                    0,
                    "save",
                    r#"{"note":"é 한","tags":[1,null],"where":{"a":true}}"#,
                )],
            ),
        ),
        (
            "a call of no arguments",
            "",
            Keep,
            format!(
                "{H}get_time<|content_invoke_tool_json|>{{\"name\":\"get_time\",\"args\":{{}}}}{E}"
            ),
            msg("", "", vec![call(0, "get_time", "{}")]),
        ),
        (
            "text outside any block is content",
            "",
            Keep,
            "Hi.<|content_model_end_sampling|>".to_owned(),
            msg("", "Hi.<|content_model_end_sampling|>", vec![]),
        ),
    ]
}

/// `(name, raw output, the error)`.
fn malformed() -> Vec<(&'static str, String, InklingError)> {
    let call = |body: &str| format!("{H}f<|content_invoke_tool_json|>{body}");
    vec![
        (
            "a name with no body at the end",
            format!("{H}get_weather"),
            InklingError::Unterminated {
                index: 0,
                text: format!("{H}get_weather"),
            },
        ),
        (
            "a header the stream ends",
            H.to_owned(),
            InklingError::Unterminated {
                index: 0,
                text: H.to_owned(),
            },
        ),
        (
            "a name that holds a <",
            format!("{H}a<b<|content_invoke_tool_json|>{{}}"),
            InklingError::NoName {
                index: 0,
                found: "a<".to_owned(),
            },
        ),
        (
            "a name that holds a newline",
            format!("{H}a\nb<|content_invoke_tool_json|>{{}}"),
            InklingError::NoName {
                index: 0,
                found: "a\n".to_owned(),
            },
        ),
        (
            "an empty name",
            format!("{H}<|content_invoke_tool_json|>{{}}"),
            InklingError::NoName {
                index: 0,
                found: String::new(),
            },
        ),
        (
            "a body that names another function",
            call(&format!("{{\"name\":\"other\",\"args\":{{}}}}{E}")),
            InklingError::BadCall {
                index: 0,
                name: "f".to_owned(),
                found: "{\"name\":\"o".to_owned(),
            },
        ),
        (
            "arguments that are no object",
            call(&format!("{{\"name\":\"f\",\"args\":[1]}}{E}")),
            InklingError::BadCall {
                index: 0,
                name: "f".to_owned(),
                found: "[1]".to_owned(),
            },
        ),
        (
            "arguments whose brackets close no JSON",
            call(&format!("{{\"name\":\"f\",\"args\":{{x}}}}{E}")),
            InklingError::BadCall {
                index: 0,
                name: "f".to_owned(),
                found: "{x}".to_owned(),
            },
        ),
        (
            "arguments that begin no JSON value",
            call("{\"name\":\"f\",\"args\":now}"),
            InklingError::BadCall {
                index: 0,
                name: "f".to_owned(),
                found: "no".to_owned(),
            },
        ),
        (
            "a body whose closing brace is missing",
            call(&format!("{{\"name\":\"f\",\"args\":{{}}{E}")),
            InklingError::BadCall {
                index: 0,
                name: "f".to_owned(),
                found: "<".to_owned(),
            },
        ),
        (
            "arguments still growing at the end",
            call("{\"name\":\"f\",\"args\":{\"a\":1"),
            InklingError::Unterminated {
                index: 0,
                text: call("{\"name\":\"f\",\"args\":{\"a\":1"),
            },
        ),
        (
            "a call whose block is not closed at the end",
            call("{\"name\":\"f\",\"args\":{\"a\":1}}"),
            InklingError::Unterminated {
                index: 0,
                text: call("{\"name\":\"f\",\"args\":{\"a\":1}}"),
            },
        ),
        (
            "the second call malformed",
            format!("{H}{CALL}{H}g<|content_invoke_tool_json|>(",),
            InklingError::BadCall {
                index: 1,
                name: "g".to_owned(),
                found: "(".to_owned(),
            },
        ),
    ]
}

fn parse(format: ReasoningFormat, pieces: &[&str]) -> Result<Message, MarkupError> {
    let mut p = ChatParser::with_tools("", format, Some(Tools::Inkling));
    common::sum_deltas(&mut p, pieces)
}

/// The fixtures parse to their messages, their arguments JSON objects; the reference's own cases
/// are the first nine.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_fixtures_parse_to_expected_messages() {
    common::check_fixtures(fixtures(), |_, format, pieces| parse(format, pieces));
}

/// Markup that does not parse is a named error, never content and never a guessed call.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_malformed_calls_are_named_errors() {
    common::check_malformed(malformed(), |pieces| {
        parse(ReasoningFormat::Deepseek, pieces)
    });
}

/// A message, or an error, never depends on how the text was cut: whole, a character at a time,
/// in two at every character, in three at every pair of tag boundaries.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_parse_does_not_depend_on_chunking() {
    common::check_chunking(
        fixtures(),
        malformed(),
        ("", ReasoningFormat::Deepseek),
        |_, format, pieces| parse(format, pieces),
    );
}

/// A call the limit cut after its name was read stays a call with the arguments read; one cut
/// before its name is the text it consumed, as the reference reads the end of the input.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_cut_call_is_kept_or_given_back() {
    let run = |text: &str| {
        let mut p = ChatParser::with_tools("", ReasoningFormat::Deepseek, Some(Tools::Inkling))
            .streaming_calls();
        p.try_push(text, &[]).unwrap();
        p.try_finish_at(true).unwrap();
        p.message().clone()
    };
    assert_eq!(
        run(&format!(
            "{H}f<|content_invoke_tool_json|>{{\"name\":\"f\",\"args\":{{\"a\":1,\"b"
        )),
        msg("", "", vec![call(0, "f", r#"{"a":1,"b"#)]),
    );
    assert_eq!(
        run(&format!("{H}f<|content_invoke_tool_js")),
        msg("", &format!("{H}f<|content_invoke_tool_js"), vec![]),
    );
}

/// The call a scan has opened is reported as it is read: its name once the body's opening is
/// whole and its first argument byte is read, the arguments' compact JSON growing to the close.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_open_calls_are_reported_as_they_are_read() {
    let mut p = ChatParser::with_tools("", ReasoningFormat::Deepseek, Some(Tools::Inkling))
        .streaming_calls();
    let mut pieces = Vec::new();
    for part in [
        format!(
            "{H}<|content_text|>On it.{E}{H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":{{\"city\": \"Par"
        ),
        "is\", \"unit\": \"C\"}".to_owned(),
        format!("}}{E}"),
    ] {
        p.try_push(&part, &[]).unwrap();
        pieces.extend(p.take_pieces());
    }
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

/// The server reads the markup from the template: both released files' templates teach Inkling's
/// blocks, and the models crate's table names it too.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_detection_follows_the_template() {
    for template in [MINI, SMALL] {
        let parsed = ChatTemplate::parse(template).expect("template parses");
        assert_eq!(ToolFormat::of_chat_template(&parsed), ToolFormat::Inkling);
        assert_eq!(ToolFormat::of_template(template), ToolFormat::Inkling);
        assert_eq!(
            models::ToolFormat::of_template(template),
            Some(models::ToolFormat::Inkling)
        );
    }
}

/// Both chat templates render as the reference's jinja engine does: every case of the fixture
/// through `/apply-template` equals its reference render byte for byte, and the cases the
/// template refuses itself (`raise_exception`) are refused naming its message. The cases reach
/// the system turn, content parts, the tool declarations (one JSON line with sorted keys), a
/// call's sorted arguments with nested and non-ASCII values, a tool result named by its call id
/// and by its name, a thinking block kept in history, and the thinking effort line.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_templates_render_as_the_reference() {
    let fixture: Value = serde_json::from_str(RENDERS).expect("fixture JSON");
    for (key, template) in [("mini", MINI), ("small", SMALL)] {
        let addr = start_templated(Box::new(ScriptedEngine::new(4096, "")), template);
        let cases = fixture["templates"][key]["cases"]
            .as_array()
            .expect("cases");
        assert_eq!(cases.len(), 12, "{key}");
        for case in cases {
            let r = post(addr, "/apply-template", &case["body"]);
            if let Some(error) = case["error"].as_str() {
                assert_ne!(r.status, 200, "{key} {}: {}", case["name"], r.body);
                assert!(r.body.contains(error), "{key} {}: {}", case["name"], r.body);
                continue;
            }
            assert_eq!(r.status, 200, "{key} {}: {}", case["name"], r.body);
            assert_eq!(r.json()["prompt"], case["prompt"], "{key} {}", case["name"]);
        }
    }
}

// ---------------------------------------------------------------- the server

fn start(script: &str, template: &str) -> std::net::SocketAddr {
    start_templated(Box::new(ScriptedEngine::new(4096, script)), template)
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object",
            "properties": {}}}},
    ])
}

fn plain_body(stream: bool) -> Value {
    json!({
        "model": "m",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0,
        "stream": stream,
    })
}

/// A reply with thinking, text and a call is split into `reasoning_content`, `content` and
/// `tool_calls`, streamed and not, on both templates; a reply with no tools asked for is split
/// the same way (the thinking block is no `<think>` span, only the scan finds it).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_reply_splits_into_blocks() {
    let reply = format!(
        "{H}<|content_thinking|>The user wants the weather.{E}{H}<|content_text|>Checking.{E}\
         {H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":\
         {{\"city\":\"Paris\",\"unit\":\"C\"}}}}{E}"
    );
    let plain = format!("{H}<|content_thinking|>Greet back.{E}{H}<|content_text|>Hello!{E}");
    for template in [MINI, SMALL] {
        for stream in [false, true] {
            let addr = start(&reply, template);
            let r = post(
                addr,
                "/v1/chat/completions",
                &common::tool_body(tools(), stream),
            );
            assert_eq!(r.status, 200, "{}", r.body);
            let (finish, message) = if stream {
                let (f, m) =
                    common::fold_stream(&r.body, &["<|message_model|>", "<|end_message|>"])
                        .expect("no error event");
                (
                    f,
                    json!({"reasoning_content": m["reasoning_content"], "content": m["content"], "calls": m["calls"]}),
                )
            } else {
                let v = r.json();
                let c = &v["choices"][0];
                let m = &c["message"];
                let calls: Vec<Value> = m["tool_calls"]
                    .as_array()
                    .expect("tool_calls")
                    .iter()
                    .map(|tc| tc["function"].clone())
                    .collect();
                (
                    c["finish_reason"].clone(),
                    json!({"reasoning_content": m["reasoning_content"], "content": m["content"], "calls": calls}),
                )
            };
            assert_eq!(finish, "tool_calls", "stream {stream}");
            assert_eq!(
                message,
                json!({
                    "reasoning_content": "The user wants the weather.",
                    "content": "Checking.",
                    "calls": [{"name": "get_weather", "arguments": r#"{"city":"Paris","unit":"C"}"#}],
                }),
                "stream {stream}"
            );

            let addr = start(&plain, template);
            let r = post(addr, "/v1/chat/completions", &plain_body(stream));
            assert_eq!(r.status, 200, "{}", r.body);
            let (reasoning, content) = if stream {
                let (_, m) =
                    common::fold_stream(&r.body, &["<|message_model|>", "<|end_message|>"])
                        .expect("no error event");
                (m["reasoning_content"].clone(), m["content"].clone())
            } else {
                let m = &r.json()["choices"][0]["message"];
                (m["reasoning_content"].clone(), m["content"].clone())
            };
            assert_eq!(
                (reasoning, content),
                (json!("Greet back."), json!("Hello!")),
                "no tools, stream {stream}"
            );
        }
    }
}

/// A request that asks for no call reads none: under `tool_choice: "none"` a reply whose last
/// block is a call keeps that block in `content` as the model wrote it, the thinking block still
/// split out and the finish `stop`, streamed and not, on both templates. Mutant: the scan that
/// reads calls under `"none"` answers a `tool_calls` entry, the finish `tool_calls` and a content
/// without the block.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_tool_choice_none_reads_no_call() {
    let call = format!(
        "{H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":\
         {{\"city\":\"Paris\"}}}}{E}"
    );
    let reply = format!(
        "{H}<|content_thinking|>The user wants the weather.{E}{H}<|content_text|>Checking.{E}{call}"
    );
    for template in [MINI, SMALL] {
        for stream in [false, true] {
            let addr = start(&reply, template);
            let mut body = common::tool_body(tools(), stream);
            body["tool_choice"] = json!("none");
            let r = post(addr, "/v1/chat/completions", &body);
            assert_eq!(r.status, 200, "{}", r.body);
            let (finish, reasoning, content, calls) = if stream {
                let (f, m) = common::fold_stream(&r.body, &[]).expect("no error event");
                (
                    f,
                    m["reasoning_content"].clone(),
                    m["content"].clone(),
                    m["calls"].as_array().expect("calls").len(),
                )
            } else {
                let v = r.json();
                let c = &v["choices"][0];
                let m = &c["message"];
                (
                    c["finish_reason"].clone(),
                    m["reasoning_content"].clone(),
                    m["content"].clone(),
                    m["tool_calls"].as_array().map_or(0, Vec::len),
                )
            };
            assert_eq!(
                (finish, reasoning, content, calls),
                (
                    json!("stop"),
                    json!("The user wants the weather."),
                    json!(format!("Checking.{call}")),
                    0
                ),
                "stream {stream}"
            );
        }
    }
}

/// Markup that does not parse is the request's named error: a 500 without a stream, an error
/// event that ends a stream. The request is unwatched ([`common::unwatched`]).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_malformed_call_is_an_api_error() {
    let script =
        format!("{H}get_time<|content_invoke_tool_json|>{{\"name\":\"other\",\"args\":{{}}}}{E}");
    common::check_api_error(
        start(&script, MINI),
        |stream| common::unwatched(common::tool_body(tools(), stream)),
        "tool-call markup: tool call 0 (get_time) is not the template's call body: \"{\\\"name\\\":\\\"o\"",
        |body| common::fold_stream(body, &[]),
    );
}
