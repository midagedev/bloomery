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
use serve::inkling::{ANSWER, InklingError};
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
            // PIN(2026-11-27): the reference renders the template alone; the server ends the
            // prompt of a request for no thinking in the answer block, which the template does not
            // write (`hw_inkling2_thinking_off_ends_the_prompt_in_the_answer_block`).
            let ends = if case["body"]["reasoning_effort"] == "none" {
                ANSWER
            } else {
                ""
            };
            let want = format!("{}{ends}", case["prompt"].as_str().expect("a prompt"));
            assert_eq!(r.json()["prompt"], json!(want), "{key} {}", case["name"]);
        }
    }
}

/// The reference's render of the case `name` of the template `key` (`mini` or `small`).
fn reference(key: &str, name: &str) -> String {
    let fixture: Value = serde_json::from_str(RENDERS).expect("fixture JSON");
    fixture["templates"][key]["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("{key}: no case {name}"))["prompt"]
        .as_str()
        .expect("a prompt")
        .to_owned()
}

/// A request for no thinking ends the prompt in the answer block: `reasoning_effort: "none"`,
/// `enable_thinking: false` (top level and in `chat_template_kwargs`), a kwargs `reasoning_effort`
/// of `"none"` and `reasoning_budget: 0` each render the template's effort-0 prompt and then
/// `<|message_model|><|content_text|>`, the template's own start of an assistant turn that is
/// only text, so the model writes no thinking block. The template writes no such prompt itself
/// (it ends at the last message and the model opens its own block, thinking among them), and a
/// request that asks for thinking or for nothing renders as the template does, byte for byte, as
/// does one that asks for no generation prompt. Mutant: a server that appends nothing renders the
/// effort-0 prompt alone for each spelling.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_thinking_off_ends_the_prompt_in_the_answer_block() {
    for (key, template) in [("mini", MINI), ("small", SMALL)] {
        let addr = start("", template);
        let prompt = |extra: &Value| -> String {
            let mut body = json!({"messages": [{"role": "user", "content": "Hello"}]});
            body.as_object_mut()
                .expect("an object")
                .extend(extra.as_object().expect("an object").clone());
            let r = post(addr, "/apply-template", &body);
            assert_eq!(r.status, 200, "{key} {extra}: {}", r.body);
            r.json()["prompt"].as_str().expect("a prompt").to_owned()
        };
        let stop = format!("{}{ANSWER}", reference(key, "effort none"));
        for extra in [
            json!({"reasoning_effort": "none"}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            json!({"enable_thinking": false}),
            json!({"chat_template_kwargs": {"reasoning_effort": "none"}}),
            json!({"reasoning_budget": 0}),
            json!({"reasoning_effort": "none", "enable_thinking": false, "reasoning_budget": 0}),
        ] {
            assert_eq!(prompt(&extra), stop, "{key} {extra}");
        }
        for (name, extra) in [
            ("a user turn", json!({})),
            ("a user turn", json!({"enable_thinking": true})),
            ("a user turn", json!({"reasoning_budget": -1})),
            ("a user turn", json!({"reasoning_budget": null})),
            ("effort low", json!({"reasoning_effort": "low"})),
            ("effort max", json!({"reasoning_effort": "max"})),
        ] {
            assert_eq!(prompt(&extra), reference(key, name), "{key} {extra}");
        }
        assert_eq!(
            prompt(&json!({"reasoning_effort": "none", "add_generation_prompt": false})),
            reference(key, "effort none"),
            "{key}: no generation prompt asked"
        );
    }
}

/// A request whose program starts at the first generated id writes no thinking and takes no
/// answer block: a forced call writes its own header and a JSON document its own text, so under
/// `reasoning_effort: "none"` they render the effort-0 prompt alone, while a `tool_choice: "auto"`
/// request, whose model is free until it opens a call, ends in the answer block. Mutant: a server
/// that appends the block whatever the program leaves a forced call's header after it.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_a_program_that_starts_at_once_gets_no_answer_block() {
    let addr = start("", MINI);
    let prompt = |extra: Value| -> String {
        let mut body = json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": tools(),
            "reasoning_effort": "none",
        });
        body.as_object_mut()
            .expect("an object")
            .extend(extra.as_object().expect("an object").clone());
        let r = post(addr, "/apply-template", &body);
        assert_eq!(r.status, 200, "{extra}: {}", r.body);
        r.json()["prompt"].as_str().expect("a prompt").to_owned()
    };
    for extra in [
        json!({"tool_choice": "required"}),
        json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
        json!({"response_format": {"type": "json_object"}}),
    ] {
        let p = prompt(extra.clone());
        assert!(p.contains("Thinking effort level: 0<"), "{extra}: {p}");
        assert!(!p.ends_with(ANSWER), "{extra}: {p}");
    }
    for extra in [json!({}), json!({"tool_choice": "auto"})] {
        let p = prompt(extra.clone());
        assert!(p.ends_with(ANSWER), "{extra}: {p}");
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

/// The message a chat answer carries as `(finish, reasoning, content, calls)`, whole or streamed.
fn message_of(r: &common::Reply, stream: bool) -> (Value, String, String, Vec<Value>) {
    assert_eq!(r.status, 200, "{}", r.body);
    if stream {
        let (finish, m) = common::fold_stream(&r.body, &[H, E]).expect("no error event");
        return (
            finish,
            m["reasoning_content"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            m["content"].as_str().unwrap_or_default().to_owned(),
            m["calls"].as_array().cloned().unwrap_or_default(),
        );
    }
    let v = r.json();
    let c = &v["choices"][0];
    let m = &c["message"];
    (
        c["finish_reason"].clone(),
        m["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        m["content"].as_str().unwrap_or_default().to_owned(),
        m["tool_calls"]
            .as_array()
            .map(|a| a.iter().map(|tc| tc["function"].clone()).collect())
            .unwrap_or_default(),
    )
}

/// A reply to a prompt that ends in the answer block is content from its first byte: the model
/// is already inside a text block, so its text is the block's and its `<|end_message|>` closes it
/// (it is no content), and a call block after it is read as ever, streamed and not. Mutant: a
/// parser that starts between blocks leaves the block's end in the content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_a_reply_in_the_answer_block_is_content() {
    let call = format!(
        "{H}get_weather<|content_invoke_tool_json|>{{\"name\":\"get_weather\",\"args\":\
         {{\"city\":\"Paris\"}}}}{E}"
    );
    for stream in [false, true] {
        let addr = start(&format!("Hello!{E}"), MINI);
        let mut body = plain_body(stream);
        body["reasoning_effort"] = json!("none");
        let (finish, reasoning, content, calls) =
            message_of(&post(addr, "/v1/chat/completions", &body), stream);
        assert_eq!(
            (finish, reasoning.as_str(), content.as_str(), calls.len()),
            (json!("stop"), "", "Hello!", 0),
            "stream {stream}"
        );

        let addr = start(&format!("On it.{E}{call}"), MINI);
        let mut body = common::tool_body(tools(), stream);
        body["reasoning_effort"] = json!("none");
        let (finish, reasoning, content, calls) =
            message_of(&post(addr, "/v1/chat/completions", &body), stream);
        assert_eq!(
            (finish, reasoning.as_str(), content.as_str()),
            (json!("tool_calls"), "", "On it."),
            "stream {stream}"
        );
        assert_eq!(
            calls,
            vec![json!({"name": "get_weather", "arguments": r#"{"city":"Paris"}"#})],
            "stream {stream}"
        );
    }
}

/// `reasoning_budget` on Inkling 2 is 0 or -1: its thinking is a block of its own that the budget
/// tracker (which reads `<think>` spans) cannot see, so a budget of 0 is the request for no
/// thinking (the prompt ends in the answer block and the reply is content), -1 and no budget leave
/// the model unrestricted, and any other budget is a 400 naming the field, on chat and on
/// `/completion`, where a prompt that already ends in the answer block takes any budget as moot.
/// Mutant: a server that accepts the budget answers 200 and never spends it.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_a_budget_is_zero_or_refused() {
    let addr = start(&format!("Hello!{E}"), MINI);
    let chat = |extra: Value| {
        let mut body = plain_body(false);
        body.as_object_mut()
            .expect("an object")
            .extend(extra.as_object().expect("an object").clone());
        post(addr, "/v1/chat/completions", &body)
    };
    let (_, reasoning, content, _) = message_of(&chat(json!({"reasoning_budget": 0})), false);
    assert_eq!((reasoning.as_str(), content.as_str()), ("", "Hello!"));
    for extra in [
        json!({"reasoning_budget": -1}),
        json!({"reasoning_budget": null}),
    ] {
        let r = chat(extra.clone());
        assert_eq!(r.status, 200, "{extra}: {}", r.body);
    }
    let r = chat(json!({"reasoning_budget": 5}));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(
        r.body.contains("reasoning_budget 5 is not supported")
            && r.body.contains("\"param\":\"reasoning_budget\""),
        "{}",
        r.body
    );
    // A program that starts at once takes no budget on any family, and a request that asks for no
    // generation prompt leaves no prompt to end in the answer block.
    for extra in [
        json!({"reasoning_budget": 0, "response_format": {"type": "json_object"}}),
        json!({"reasoning_budget": 0, "tools": tools(), "tool_choice": "required"}),
    ] {
        let r = chat(extra.clone());
        assert_eq!(r.status, 400, "{extra}: {}", r.body);
        assert!(
            r.body
                .contains("reasoning_budget is not supported with structured output"),
            "{extra}: {}",
            r.body
        );
    }
    let r = chat(json!({"reasoning_budget": 0, "add_generation_prompt": false}));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(
        r.body.contains("reasoning_budget 0 is not supported")
            && r.body
                .contains("a chat request that asks for a generation prompt"),
        "{}",
        r.body
    );
    let raw = |prompt: &str| {
        post(
            addr,
            "/completion",
            &json!({"prompt": prompt, "n_predict": 4, "reasoning_budget": 3}),
        )
    };
    let r = raw("<|message_user|><|content_text|>Hi<|end_message|>");
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(
        r.body.contains("reasoning_budget 3 is not supported"),
        "{}",
        r.body
    );
    let r = raw(&format!(
        "<|message_user|><|content_text|>Hi<|end_message|>{ANSWER}"
    ));
    assert_eq!(r.status, 200, "{}", r.body);
}

/// `/v1/messages/count_tokens`' `input_tokens` for a request body whose `extra` fields are added
/// to one user message.
fn counted(addr: std::net::SocketAddr, extra: Value) -> u64 {
    let mut body = json!({"messages": [{"role": "user", "content": "Hello"}]});
    body.as_object_mut()
        .expect("an object")
        .extend(extra.as_object().expect("an object").clone());
    let r = post(addr, "/v1/messages/count_tokens", &body);
    assert_eq!(r.status, 200, "{extra}: {}", r.body);
    r.json()["input_tokens"].as_u64().expect("input_tokens")
}

/// Anthropic's `thinking: {"type": "disabled"}` is thinking off like the chat path's other
/// switches: the prompt it renders is the one `chat_template_kwargs.enable_thinking: false`
/// renders (the effort-0 prompt that ends in the answer block, counted by `count_tokens`), the
/// reply is a text block from its first byte with no thinking block before it, and one that
/// also sends `enable_thinking: true` in the kwargs is a 400 naming both. Mutant: a conversion
/// that drops `disabled` leaves the prompt as it was, so the count moves and the block's end
/// stays in the reply's text.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_messages_thinking_disabled_is_thinking_off() {
    let addr = start(&format!("Hello!{E}"), MINI);
    let ask = |extra: Value| {
        let mut body =
            json!({"max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}]});
        body.as_object_mut()
            .expect("an object")
            .extend(extra.as_object().expect("an object").clone());
        post(addr, "/v1/messages", &body)
    };
    let r = ask(json!({"thinking": {"type": "disabled"}}));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(
        r.json()["content"],
        json!([{"type": "text", "text": "Hello!"}])
    );
    let off = counted(addr, json!({"thinking": {"type": "disabled"}}));
    assert_eq!(
        off,
        counted(
            addr,
            json!({"chat_template_kwargs": {"enable_thinking": false}})
        )
    );
    assert!(off > counted(addr, json!({})), "{off}");
    assert_eq!(
        counted(addr, json!({"thinking": {"type": "adaptive"}})),
        counted(addr, json!({}))
    );
    let r = ask(json!({
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"enable_thinking": true},
    }));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(
        r.body.contains("thinking.type is disabled") && r.body.contains("opposite thinking"),
        "{}",
        r.body
    );
}

/// `thinking: {"type": "enabled", "budget_tokens": N}` on Inkling 2 is not refused: clients send
/// it on every request, and no tracker closes this model's thinking block, so the request runs
/// with the thinking the template has, unbounded (the thinking block and the text block both
/// come back), and the prompt is the one a request with no `thinking` renders. Mutant: a
/// conversion that keeps the budget is the chat path's 400.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_inkling2_messages_enabled_budget_is_not_refused() {
    let reply = format!("{H}<|content_thinking|>Greet back.{E}{H}<|content_text|>Hello!{E}");
    let addr = start(&reply, MINI);
    for budget in [2048, 1] {
        let r = post(
            addr,
            "/v1/messages",
            &json!({
                "max_tokens": 400,
                "messages": [{"role": "user", "content": "Hello"}],
                "thinking": {"type": "enabled", "budget_tokens": budget},
            }),
        );
        assert_eq!(r.status, 200, "{budget}: {}", r.body);
        assert_eq!(
            r.json()["content"],
            json!([
                {"type": "thinking", "thinking": "Greet back.", "signature": ""},
                {"type": "text", "text": "Hello!"},
            ]),
            "{budget}"
        );
    }
    assert_eq!(
        counted(
            addr,
            json!({"thinking": {"type": "enabled", "budget_tokens": 2048}})
        ),
        counted(addr, json!({}))
    );
}
