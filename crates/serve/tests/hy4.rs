//! Gate: Hy4-flash's chat in `/v1/chat/completions` and `/apply-template`. The model's template
//! (`tests/fixtures/hy4-chat-template.jinja`, the GGUF's `tokenizer.chat_template`) builds every tag
//! from one suffix, `'<tool_calls{}>'.format(HYTK)` with `HYTK = ':6124c78e'`, and writes all the calls of
//! a message between one group's tags:
//!
//! ```text
//! <tool_calls:H><tool_call:H>NAME<arg_key:H>KEY</arg_key:H><arg_value:H>VALUE</arg_value:H>…</tool_call:H>…</tool_calls:H>
//! ```
//!
//! and its generation prompt ends in the open span `<think:H>` (the closed pair under `no_think`).
//! `KvSpelling::Hy4` is the key/value scan of GLM and Solar with those tags as data and the group
//! around the calls; the server finds it by the call the template renders, since no tag is spelled
//! whole in its source. In the fixtures `~` stands for the suffix.
//!
//! The template renders as jinja2 does (`hw_hy4_template_renders_as_jinja2`), the model's calls read back
//! to messages however the stream is cut, and the calls read back render through the template to the
//! markup they were read from (`hw_hy4_calls_round_trip_through_the_template`).

mod common;

use common::{kv_scan as scan, kv_tools as tools, msg, post, tool_call as call};
use serde_json::{Map, Value, json};
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, Spelling, ToolFormat};
use serve::glmxml::{KvError, KvSpelling};
use serve::reasoning::{ReasoningFormat, ThinkTags};
use serve::{ChatTemplate, ScriptedEngine};

const TEMPLATE: &str = include_str!("fixtures/hy4-chat-template.jinja");

/// The template's reasoning span: the generation prompt ends in the open tag.
const THINK: ThinkTags = ThinkTags {
    open: "<think:6124c78e>",
    close: "</think:6124c78e>",
    response: None,
};

const GROUP_OPEN: &str = "<tool_calls:6124c78e>";
const GROUP_CLOSE: &str = "</tool_calls:6124c78e>";
const CALL_OPEN: &str = "<tool_call:6124c78e>";
const CALL_CLOSE: &str = "</tool_call:6124c78e>";

/// `s` with the suffix spelled where `~` stands.
fn hy(s: &str) -> String {
    s.replace('~', ":6124c78e")
}

/// The template's generation prompt after one user message `hi`, with the span open.
fn on() -> String {
    hy(
        "<｜hy_start~｜>system<｜hy_middle~｜><｜reasoning_mode~｜>reasoning_effort:high<｜hy_end~｜><｜hy_start~｜>user<｜hy_middle~｜>hi<｜hy_end~｜><｜hy_start~｜>assistant<｜hy_middle~｜><think~>",
    )
}

/// A prompt whose span is already closed (`no_think`).
fn closed() -> String {
    hy(
        "<｜hy_start~｜>system<｜hy_middle~｜><｜reasoning_mode~｜>reasoning_effort:no_think<｜hy_end~｜><｜hy_start~｜>user<｜hy_middle~｜>hi<｜hy_end~｜><｜hy_start~｜>assistant<｜hy_middle~｜><think~></think~>",
    )
}

const ONE_CALL: &str = "Need the weather.</think~><tool_calls~><tool_call~>get_weather<arg_key~>location</arg_key~><arg_value~>Seoul</arg_value~><arg_key~>days</arg_key~><arg_value~>3</arg_value~></tool_call~></tool_calls~>";
const TWO_CALLS: &str = "Both.</think~>Checking both.<tool_calls~><tool_call~>get_weather<arg_key~>location</arg_key~><arg_value~>Seoul</arg_value~></tool_call~><tool_call~>get_time</tool_call~></tool_calls~>";
/// A JSON value, and a string that holds a `<`, a quote and the close tag spelled without the
/// suffix and with another one.
const NOTE: &str = "</think~><tool_calls~><tool_call~>save_note<arg_key~>filter</arg_key~><arg_value~>{\"n\": [1, 2], \"q\": \"서울 ☃\"}</arg_value~><arg_key~>text</arg_key~><arg_value~>a<b & \"c\" </arg_value and </arg_value:6124c78f></arg_value~></tool_call~></tool_calls~>";
const TYPED: &str = "</think~><tool_calls~><tool_call~>get_weather<arg_key~>location</arg_key~><arg_value~>123</arg_value~><arg_key~>days</arg_key~><arg_value~>3</arg_value~></tool_call~></tool_calls~>";
const UNTYPED: &str = "</think~><tool_calls~><tool_call~>lookup<arg_key~>q</arg_key~><arg_value~>42</arg_value~><arg_key~>w</arg_key~><arg_value~>word</arg_value~><arg_key~>o</arg_key~><arg_value~>{\"a\": null}</arg_value~></tool_call~></tool_calls~>";
/// Whitespace between the group's tags is no content.
const SPACED: &str = "</think~><tool_calls~>\n<tool_call~>get_time</tool_call~>\n <tool_call~>get_time</tool_call~>\n</tool_calls~>";
/// The stream ends after the calls, before the group's close.
const NO_CLOSE: &str = "</think~><tool_calls~><tool_call~>get_time</tool_call~>";
/// A call with no group around it is a call.
const BARE: &str = "</think~><tool_call~>get_time</tool_call~>";
/// Two groups, content between them: the calls are numbered across the message.
const TWO_GROUPS: &str = "</think~>Sure.<tool_calls~><tool_call~>get_time</tool_call~></tool_calls~> and <tool_calls~><tool_call~>get_time</tool_call~></tool_calls~>";
const IN_REASONING: &str = "I could call <tool_calls~><tool_call~>get_time</tool_call~></tool_calls~> or <tool_call~>bad</arg_key~> but no.</think~>Done.";
const FORMAT_NONE: &str =
    "Think.</think~><tool_calls~><tool_call~>get_time</tool_call~></tool_calls~>";
const AFTER_CLOSED: &str = "<tool_calls~><tool_call~>get_time</tool_call~></tool_calls~> after";
const ANSWER: &str = "Hi.</think~>Hello!";
const NEAR_TAG: &str = "</think~>a <tool_calls b <tool_calls~";
/// The tags without the suffix are text: the suffix is the model's, not the markup's.
const PLAIN_TAGS: &str = "</think~>Use <tool_calls><tool_call>x</tool_call></tool_calls> to call.";

/// `(name, prompt, format, raw output, expected message)`.
fn fixtures() -> Vec<(&'static str, String, ReasoningFormat, String, Message)> {
    use ReasoningFormat::{Deepseek, None};
    let time = |i| call(i, "get_time", "{}");
    vec![
        (
            "the template's example, one call",
            on(),
            Deepseek,
            hy(ONE_CALL),
            msg(
                "Need the weather.",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            ),
        ),
        (
            "two calls in one group",
            on(),
            Deepseek,
            hy(TWO_CALLS),
            msg(
                "Both.",
                "Checking both.",
                vec![call(0, "get_weather", r#"{"location":"Seoul"}"#), time(1)],
            ),
        ),
        (
            "a JSON-valued argument, a string with < and near-miss close tags",
            on(),
            Deepseek,
            hy(NOTE),
            msg(
                "",
                "",
                vec![call(
                    0,
                    "save_note",
                    r#"{"filter":{"n":[1,2],"q":"서울 ☃"},"text":"a<b & \"c\" </arg_value and </arg_value:6124c78f>"}"#,
                )],
            ),
        ),
        (
            "a string-typed value that reads as JSON stays text",
            on(),
            Deepseek,
            hy(TYPED),
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"123","days":3}"#)],
            ),
        ),
        (
            "an unknown function's values: JSON, else text",
            on(),
            Deepseek,
            hy(UNTYPED),
            msg(
                "",
                "",
                vec![call(0, "lookup", r#"{"q":42,"w":"word","o":{"a":null}}"#)],
            ),
        ),
        (
            "whitespace between the group's tags",
            on(),
            Deepseek,
            hy(SPACED),
            msg("", "", vec![time(0), time(1)]),
        ),
        (
            "a group the stream ends inside, after its calls",
            on(),
            Deepseek,
            hy(NO_CLOSE),
            msg("", "", vec![time(0)]),
        ),
        (
            "a call with no group",
            on(),
            Deepseek,
            hy(BARE),
            msg("", "", vec![time(0)]),
        ),
        (
            "two groups, numbered across the message",
            on(),
            Deepseek,
            hy(TWO_GROUPS),
            msg("", "Sure. and ", vec![time(0), time(1)]),
        ),
        (
            "markup inside the think span is reasoning",
            on(),
            Deepseek,
            hy(IN_REASONING),
            msg(
                &hy(
                    "I could call <tool_calls~><tool_call~>get_time</tool_call~></tool_calls~> or <tool_call~>bad</arg_key~> but no.",
                ),
                "Done.",
                vec![],
            ),
        ),
        (
            "reasoning_format none keeps the span in content",
            on(),
            None,
            hy(FORMAT_NONE),
            msg("", &hy("Think.</think~>"), vec![time(0)]),
        ),
        (
            "a span the prompt closed",
            closed(),
            Deepseek,
            hy(AFTER_CLOSED),
            msg("", " after", vec![time(0)]),
        ),
        (
            "no call",
            on(),
            Deepseek,
            hy(ANSWER),
            msg("Hi.", "Hello!", vec![]),
        ),
        (
            "text that is not a tag, and a tag cut short at the end",
            on(),
            Deepseek,
            hy(NEAR_TAG),
            msg("", &hy("a <tool_calls b <tool_calls~"), vec![]),
        ),
        (
            "the tags without the suffix are text",
            on(),
            Deepseek,
            hy(PLAIN_TAGS),
            msg(
                "",
                "Use <tool_calls><tool_call>x</tool_call></tool_calls> to call.",
                vec![],
            ),
        ),
    ]
}

/// `(name, raw output, the error)`, each after the prompt [`on`].
fn malformed() -> Vec<(&'static str, String, KvError)> {
    let unexpected = |index, name: &str, expected: &str, found: &str| KvError::Unexpected {
        index,
        name: name.to_owned(),
        expected: expected.to_owned(),
        found: found.to_owned(),
    };
    let between = format!("{CALL_OPEN} or {GROUP_CLOSE}");
    vec![
        (
            "a group with no call",
            hy("</think~><tool_calls~>"),
            KvError::Unterminated {
                index: 0,
                close: GROUP_CLOSE,
                text: GROUP_OPEN.to_owned(),
            },
        ),
        (
            "a call unterminated in a group",
            hy("</think~><tool_calls~><tool_call~>get_time<arg_key~>x</arg_key~>"),
            KvError::Unterminated {
                index: 0,
                close: CALL_CLOSE,
                text: hy("<tool_call~>get_time<arg_key~>x</arg_key~>"),
            },
        ),
        (
            "the group's close where the call's belongs",
            hy("</think~><tool_calls~><tool_call~>get_time</tool_calls~>"),
            KvError::Unterminated {
                index: 0,
                close: CALL_CLOSE,
                text: hy("<tool_call~>get_time</tool_calls~>"),
            },
        ),
        (
            "text in a group before its first call",
            hy("</think~><tool_calls~>oops"),
            KvError::GroupText {
                index: 0,
                expected: between.clone(),
                found: "oops".to_owned(),
            },
        ),
        (
            "text between the calls",
            hy("</think~><tool_calls~><tool_call~>get_time</tool_call~>junk</tool_calls~>"),
            KvError::GroupText {
                index: 1,
                expected: between.clone(),
                found: hy("junk</tool_calls~>"),
            },
        ),
        (
            "text between the calls, longer than the error's quote",
            hy(
                "</think~><tool_calls~><tool_call~>get_time</tool_call~>0123456789 0123456789 0123456789 0123456789 more</tool_calls~>",
            ),
            KvError::GroupText {
                index: 1,
                expected: between,
                found: "0123456789 0123456789 0123456789 0123456".to_owned(),
            },
        ),
        (
            "a group cut inside a tag after its calls",
            hy("</think~><tool_calls~><tool_call~>get_time</tool_call~><tool_c"),
            KvError::Unterminated {
                index: 1,
                close: GROUP_CLOSE,
                text: "<tool_c".to_owned(),
            },
        ),
        (
            "arg_value without arg_key",
            hy(
                "</think~><tool_calls~><tool_call~>get_time<arg_value~>1</arg_value~></tool_call~></tool_calls~>",
            ),
            unexpected(
                0,
                "get_time",
                &format!("{} or {CALL_CLOSE}", "<arg_key:6124c78e>"),
                // The first 40 characters of the text where a key's tag belongs.
                "<arg_value:6124c78e>1</arg_value:6124c78",
            ),
        ),
        (
            "the second call malformed",
            hy(
                "</think~><tool_calls~><tool_call~>get_time</tool_call~><tool_call~>get_time<arg_value~>1</arg_value~></tool_call~></tool_calls~>",
            ),
            unexpected(
                1,
                "get_time",
                &format!("{} or {CALL_CLOSE}", "<arg_key:6124c78e>"),
                // The first 40 characters of the text where a key's tag belongs.
                "<arg_value:6124c78e>1</arg_value:6124c78",
            ),
        ),
        (
            "a key given twice",
            hy(
                "</think~><tool_calls~><tool_call~>f<arg_key~>a</arg_key~><arg_value~>1</arg_value~><arg_key~>a</arg_key~><arg_value~>2</arg_value~></tool_call~></tool_calls~>",
            ),
            KvError::DuplicateKey {
                index: 0,
                name: "f".to_owned(),
                key: "a".to_owned(),
            },
        ),
        (
            "an integer argument that is not JSON",
            hy(
                "</think~><tool_calls~><tool_call~>get_weather<arg_key~>days</arg_key~><arg_value~>three</arg_value~></tool_call~></tool_calls~>",
            ),
            KvError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            },
        ),
    ]
}

/// Feeds `pieces` one by one on the span's tags; the deltas' sum, checked against the message.
fn streamed(
    prompt: &str,
    format: ReasoningFormat,
    pieces: &[&str],
) -> Result<Message, MarkupError> {
    streamed_as(KvSpelling::Hy4, prompt, format, pieces)
}

/// [`streamed`] under `spelling`'s scan.
fn streamed_as(
    spelling: KvSpelling,
    prompt: &str,
    format: ReasoningFormat,
    pieces: &[&str],
) -> Result<Message, MarkupError> {
    let mut p = ChatParser::with_tags(prompt, format, scan(spelling), THINK);
    common::sum_deltas(&mut p, pieces)
}

#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_fixtures_parse_to_expected_messages() {
    common::check_fixtures(fixtures(), streamed);
}

#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_malformed_calls_are_named_errors() {
    common::check_malformed(malformed(), |pieces| {
        streamed(&on(), ReasoningFormat::Deepseek, pieces)
    });
}

/// A message, or an error, never depends on how the text was cut: whole, a
/// character at a time, in two at every character, in three at every pair
/// of tag boundaries (a call split across three stream pieces at its tags).
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_parse_does_not_depend_on_chunking() {
    let on = on();
    common::check_chunking(
        fixtures(),
        malformed(),
        (&on, ReasoningFormat::Deepseek),
        streamed,
    );
}

/// A streaming Hy4 parser over `pieces`: the message, and the call pieces it reported.
fn call_pieces(pieces: &[&str]) -> (Message, Vec<CallPiece>) {
    let mut p = ChatParser::with_tags(
        &on(),
        ReasoningFormat::Deepseek,
        scan(KvSpelling::Hy4),
        THINK,
    )
    .streaming_calls();
    let mut out = Vec::new();
    for piece in pieces {
        p.try_push(piece, &[]).expect("parses");
        out.extend(p.take_pieces());
    }
    p.try_finish().expect("closes");
    out.extend(p.take_pieces());
    (p.message().clone(), out)
}

/// Under every cut a group's calls stream the arguments they close with: the pieces concatenate to
/// the calls the parser that does not stream reads, and the group's tags and the whitespace between
/// its calls reach no fragment.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_every_cut_streams_the_arguments_the_call_closes_with() {
    for raw in [hy(TWO_CALLS), hy(SPACED), hy(NOTE), hy(TWO_GROUPS)] {
        for pieces in common::every_chunking(&raw, 3) {
            let (got, streamed) = call_pieces(&pieces);
            assert!(!got.calls.is_empty(), "{raw}");
            let mut calls: Vec<(String, String)> = Vec::new();
            for piece in &streamed {
                match piece {
                    CallPiece::Open { index, name } => {
                        assert_eq!(*index, calls.len(), "calls open in order: {pieces:?}");
                        calls.push((name.clone(), String::new()));
                    }
                    CallPiece::Args { index, text } => calls[*index].1.push_str(text),
                }
            }
            let closed: Vec<(String, String)> = got
                .calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect();
            assert_eq!(calls, closed, "{pieces:?}");
        }
    }
}

/// The group's tags are markup of this spelling alone: another family's scan reads Hy4's call as
/// text, and Hy4's scan reads GLM's and Solar's as text.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_and_the_other_spellings_read_only_their_own_markers() {
    const GLM_CALL: &str =
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Seoul</arg_value></tool_call>";
    const SOLAR_CALL: &str = "<|tool_call:start|>get_time\n<|tool_call:end|>\n";
    let hy_call = hy(ONE_CALL.split_once("</think~>").expect("a span").1);
    for (spelling, raw) in [
        (KvSpelling::Hy4, GLM_CALL),
        (KvSpelling::Hy4, SOLAR_CALL),
        (KvSpelling::Glm, hy_call.as_str()),
        (KvSpelling::Solar, hy_call.as_str()),
    ] {
        let got = streamed_as(spelling, "", ReasoningFormat::None, &[raw]).expect("no markup");
        assert_eq!(got, msg("", raw, vec![]), "{spelling:?}");
    }
}

/// The calls parsed from the fixtures, rendered back through the template as an assistant turn, are
/// the fixtures' markup byte for byte: the parser's text/JSON rule is the template's
/// `tojson if v is not string` read back, and the group is the template's one group a message.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_calls_round_trip_through_the_template() {
    let t = ChatTemplate::parse(TEMPLATE).expect("template");
    for raw in [hy(TWO_CALLS), hy(NOTE), hy(UNTYPED), hy(TYPED)] {
        let m = streamed(&on(), ReasoningFormat::Deepseek, &[&raw]).expect("parses");
        assert!(
            !m.calls.is_empty(),
            "the round trip goes through parsed calls: {m:?}"
        );
        let calls: Vec<Value> = m
            .calls
            .iter()
            .map(|c| {
                let args: Value = serde_json::from_str(&c.arguments).expect("JSON");
                json!({"function": {"arguments": args, "name": c.name}, "type": "function"})
            })
            .collect();
        let Value::Object(vars) = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": m.content, "tool_calls": calls},
            ],
            "add_generation_prompt": false,
        }) else {
            unreachable!("an object")
        };
        let out = t.render(&vars).expect("render");
        let (_, after) = raw.split_once(&hy("</think~>")).expect("a closed span");
        let want = hy(&format!(
            "<｜hy_start~｜>assistant<｜hy_middle~｜><think~></think~>{after}<｜hy_end~｜>"
        ));
        assert!(out.ends_with(&want), "{out}\n  does not end with\n{want}");
    }
}

/// The template's own generation prompts are the ones the fixtures follow.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_generation_prompts_are_the_fixtures() {
    let t = ChatTemplate::parse(TEMPLATE).expect("template");
    for (effort, want) in [(json!(null), on()), (json!("no_think"), closed())] {
        let mut vars = Map::new();
        vars.insert(
            "messages".into(),
            json!([{"role": "user", "content": "hi"}]),
        );
        vars.insert("add_generation_prompt".into(), json!(true));
        if !effort.is_null() {
            vars.insert("reasoning_effort".into(), effort);
        }
        assert_eq!(t.render(&vars).expect("render"), want);
    }
}

/// The server reads the markup from the call its template renders: no tag is spelled whole in the
/// source, so the source alone names no parser, and the other templates stay theirs.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_tool_format_follows_the_template() {
    assert_eq!(ToolFormat::of_template(TEMPLATE), ToolFormat::Unparsed);
    let parsed = ChatTemplate::parse(TEMPLATE).expect("template parses");
    assert_eq!(
        ToolFormat::of_chat_template(&parsed),
        ToolFormat::KvMarkup(KvSpelling::Hy4)
    );
    for (template, format) in [
        (
            include_str!("fixtures/glm5-chat-template.jinja"),
            ToolFormat::KvMarkup(KvSpelling::Glm),
        ),
        (common::V41_TEMPLATE, ToolFormat::Dsml(Spelling::Joined)),
        (
            include_str!("fixtures/qwen3-chat-template.jinja"),
            ToolFormat::Hermes,
        ),
    ] {
        let parsed = ChatTemplate::parse(template).expect("template parses");
        assert_eq!(ToolFormat::of_chat_template(&parsed), format);
    }
}

/// `body` with each tool call's `arguments` that is a JSON string parsed to the object it encodes:
/// the body a template that walks `arguments` as an object renders, as the server hands it to one.
fn arguments_parsed(body: &Value) -> Value {
    let mut body = body.clone();
    for message in body["messages"].as_array_mut().expect("messages") {
        for call in message["tool_calls"].as_array_mut().into_iter().flatten() {
            let text = call["function"]["arguments"].as_str().map(str::to_owned);
            if let Some(text) = text {
                call["function"]["arguments"] = serde_json::from_str(&text).expect("JSON");
            }
        }
    }
    body
}

/// Hy4-flash's chat template renders as jinja2 does: every case of the fixture through
/// `/apply-template` equals its reference render byte for byte. The case jinja2 refuses (tool-call
/// `arguments` given as a JSON string, which the template walks with `.items()`) renders as the same
/// body with the string parsed does, as llama-server hands such a template the object; the case the
/// template itself refuses (`raise_exception` on an effort outside its list) is a 400 naming it. The
/// cases reach the 22 `.format()` calls of the tags, `reasoning_effort`'s list, `no_think`, the
/// preserved thinking of the turns after the last user, `raw_last_assistant`, the tool results'
/// grouping, `tojson(ensure_ascii=False)` and the tools header's `defer_loading` filter.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_template_renders_as_jinja2() {
    let addr = common::start_templated(Box::new(serve::MockEngine::new(4096)), TEMPLATE);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/hy4-renders.json")).expect("fixture JSON");
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 25, "{fixture}");
    for case in cases {
        let name = &case["name"];
        let r = post(addr, "/apply-template", &case["body"]);
        match case["error"].as_str() {
            Some(e) if e.starts_with("UndefinedError: ") => {
                // PIN(2026-11-18): a string `arguments` is parsed for a template that walks it, no longer refused by name.
                assert_eq!(r.status, 200, "{name}: {}", r.body);
                let parsed = post(addr, "/apply-template", &arguments_parsed(&case["body"]));
                assert_eq!(parsed.status, 200, "{name}: {}", parsed.body);
                assert_eq!(r.json()["prompt"], parsed.json()["prompt"], "{name}");
                assert!(
                    r.json()["prompt"]
                        .as_str()
                        .is_some_and(|p| p.contains("get_time")),
                    "{name}: {}",
                    r.body
                );
            }
            Some(e) => {
                let want = e.strip_prefix("TemplateError: ").expect("raise_exception");
                assert_eq!(r.status, 400, "{name}: {}", r.body);
                assert!(r.body.contains(want), "{name}: {}", r.body);
            }
            None => {
                assert_eq!(r.status, 200, "{name}: {}", r.body);
                assert_eq!(r.json()["prompt"], case["prompt"], "{name}");
            }
        }
    }
}

// ---------------------------------------------------------------- the server

fn start(script: &str) -> std::net::SocketAddr {
    common::start_tagged(Box::new(ScriptedEngine::new(4096, script)), TEMPLATE, THINK)
}

/// A streamed chat body folded back ([`common::fold_stream`]) under this spelling's markup.
fn fold_stream(body: &str) -> Result<(Value, Value), String> {
    common::fold_stream(body, &["<tool_call"])
}

/// A Hy4 reply through `/v1/chat/completions`, streamed and not: the reasoning is split at the
/// template's span tags, the group's tags are never content, and the streamed calls fold to the
/// non-streamed ones.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_chat_tool_calls_stream_and_non_stream() {
    let addr = start(&hy(TWO_CALLS.replace("Both.", "Need both.").as_str()));
    let r = post(
        addr,
        "/v1/chat/completions",
        &common::tool_body(tools(), false),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["reasoning_content"], "Need both.", "{v}");
    assert_eq!(c["message"]["content"], "Checking both.", "{v}");
    let calls: Vec<Value> = c["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(
        calls,
        [
            json!({"name": "get_weather", "arguments": r#"{"location":"Seoul"}"#}),
            json!({"name": "get_time", "arguments": "{}"}),
        ]
    );
    let r = post(
        addr,
        "/v1/chat/completions",
        &common::tool_body(tools(), true),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["reasoning_content"], "Need both.");
    assert_eq!(folded["content"], "Checking both.");
    assert_eq!(folded["calls"], json!(calls), "streamed == non-streamed");
}

/// Markup that does not parse is the request's named error: a 500 without a stream, an error event
/// that ends a stream (no `[DONE]`), for a malformed call, for a group with text in it and for one
/// the generation leaves open.
#[test]
#[ignore = "gate: just gate-hy4-tokenizer"]
fn hw_hy4_malformed_call_is_an_api_error() {
    for (script, want) in [
        (
            hy(
                "</think~><tool_calls~><tool_call~>get_time<arg_value~>1</arg_value~></tool_call~></tool_calls~>",
            ),
            "tool-call markup: tool call 0 (get_time): expected <arg_key:6124c78e> or </tool_call:6124c78e>",
        ),
        (
            hy("</think~>Calling.<tool_calls~><tool_call~>get_time<arg_key~>x"),
            "tool-call markup: tool call 0 is not closed",
        ),
        (
            hy("</think~><tool_calls~><tool_call~>get_time</tool_call~>junk</tool_calls~>"),
            "tool-call markup: tool calls group: expected <tool_call:6124c78e> or </tool_calls:6124c78e> after call 1",
        ),
    ] {
        common::check_api_error(
            start(&script),
            |stream| common::unwatched(common::tool_body(tools(), stream)),
            want,
            fold_stream,
        );
    }
}
