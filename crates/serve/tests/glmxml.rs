//! Gate: GLM tool calls in `/v1/chat/completions` — the markup of the
//! GLM-5.3-Flash chat template (`tests/fixtures/glm5-chat-template.jinja`, the
//! GGUF's `tokenizer.chat_template`), its line 50 and its assistant branch:
//!
//! ```text
//! <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>
//! {{- '<tool_call>' + tc.name -}}
//! {% for k, v in _args.items() %}<arg_key>{{ k }}</arg_key><arg_value>{{ v | tojson(ensure_ascii=False) if v is not string else v }}</arg_value>{% endfor %}</tool_call>
//! ```
//!
//! and its generation prompt, which always ends `<|assistant|><think>`: the
//! model's output starts inside the think span, whose `</think>` rule is
//! `reasoning.rs`'s. `hw_glm_fixtures_are_the_templates_markup` renders the
//! parsed calls back through the template and gets the fixture text again.

mod common;

use common::post;
use serde_json::{Map, Value, json};
use serve::dsml::{ChatParser, Message, ToolCall, ToolFormat, Tools};
use serve::glmxml::{ArgTypes, GlmXmlError};
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const TEMPLATE: &str = include_str!("fixtures/glm5-chat-template.jinja");

/// The template's generation prompt after one user turn.
const ON: &str = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>";
/// A prompt whose think span is already closed.
const CLOSED: &str =
    "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think></think>";

const ONE_CALL: &str = "Need the weather.</think><tool_call>get_weather<arg_key>location</arg_key><arg_value>Seoul</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
const TWO_CALLS: &str = "Both.</think>Checking both.<tool_call>get_weather<arg_key>location</arg_key><arg_value>Seoul</arg_value></tool_call><tool_call>get_time</tool_call>";
const NOTE: &str = "</think><tool_call>save_note<arg_key>filter</arg_key><arg_value>{\"n\": [1, 2], \"q\": \"서울 ☃\"}</arg_value><arg_key>text</arg_key><arg_value>a<b & \"c\" </tool</arg_value></tool_call>";
const TYPED: &str = "</think><tool_call>get_weather<arg_key>location</arg_key><arg_value>123</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
const UNTYPED: &str = "</think><tool_call>lookup<arg_key>q</arg_key><arg_value>42</arg_value><arg_key>w</arg_key><arg_value>word</arg_value><arg_key>o</arg_key><arg_value>{\"a\": null}</arg_value></tool_call>";
const NEWLINES: &str = "</think><tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Seoul</arg_value>\n</tool_call>";
const IN_REASONING: &str =
    "I could call <tool_call>get_time</tool_call> or <tool_call>bad</arg_key> but no.</think>Done.";
const FORMAT_NONE: &str = "Think.</think><tool_call>get_time</tool_call>";
const AFTER_CLOSED: &str = "<tool_call>get_time</tool_call> after";
const ANSWER: &str = "Hi.</think>Hello!";
const NEAR_TAG: &str = "</think>a <tool_ca b <tool_call";

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "save_note", "parameters": {"type": "object",
            "properties": {"filter": {"type": "object"}, "text": {"type": "string"}}}}},
    ])
}

fn glm() -> Option<Tools> {
    Some(Tools::GlmXml(ArgTypes::of_tools(Some(&tools()))))
}

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
    use ReasoningFormat::{Deepseek, None};
    vec![
        (
            "the template's example, one call",
            ON,
            Deepseek,
            ONE_CALL,
            msg(
                "Need the weather.",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            ),
        ),
        (
            "two calls in one message",
            ON,
            Deepseek,
            TWO_CALLS,
            msg(
                "Both.",
                "Checking both.",
                vec![
                    call(0, "get_weather", r#"{"location":"Seoul"}"#),
                    call(1, "get_time", "{}"),
                ],
            ),
        ),
        (
            "a JSON-valued argument, a string with < and a partial close tag",
            ON,
            Deepseek,
            NOTE,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "save_note",
                    r#"{"filter":{"n":[1,2],"q":"서울 ☃"},"text":"a<b & \"c\" </tool"}"#,
                )],
            ),
        ),
        (
            "a string-typed value that reads as JSON stays text",
            ON,
            Deepseek,
            TYPED,
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"123","days":3}"#)],
            ),
        ),
        (
            "an unknown function's values: JSON, else text",
            ON,
            Deepseek,
            UNTYPED,
            msg(
                "",
                "",
                vec![call(0, "lookup", r#"{"q":42,"w":"word","o":{"a":null}}"#)],
            ),
        ),
        (
            "newlines between tags",
            ON,
            Deepseek,
            NEWLINES,
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul"}"#)],
            ),
        ),
        (
            "markup inside the think span is reasoning",
            ON,
            Deepseek,
            IN_REASONING,
            msg(
                "I could call <tool_call>get_time</tool_call> or <tool_call>bad</arg_key> but no.",
                "Done.",
                vec![],
            ),
        ),
        (
            "reasoning_format none keeps the span in content",
            ON,
            None,
            FORMAT_NONE,
            msg("", "Think.</think>", vec![call(0, "get_time", "{}")]),
        ),
        (
            "a span the prompt closed",
            CLOSED,
            Deepseek,
            AFTER_CLOSED,
            msg("", " after", vec![call(0, "get_time", "{}")]),
        ),
        (
            "no call",
            ON,
            Deepseek,
            ANSWER,
            msg("Hi.", "Hello!", vec![]),
        ),
        (
            "text that is not a call opening, and one cut short at the end",
            ON,
            Deepseek,
            NEAR_TAG,
            msg("", "a <tool_ca b <tool_call", vec![]),
        ),
    ]
}

/// `(name, raw output, the error)`, each after the prompt [`ON`].
fn malformed() -> Vec<(&'static str, &'static str, GlmXmlError)> {
    let unexpected = |index, name: &str, expected, found: &str| GlmXmlError::Unexpected {
        index,
        name: name.to_owned(),
        expected,
        found: found.to_owned(),
    };
    vec![
        (
            "unterminated",
            "</think><tool_call>get_time<arg_key>x</arg_key>",
            GlmXmlError::Unterminated {
                index: 0,
                text: "<tool_call>get_time<arg_key>x</arg_key>".to_owned(),
            },
        ),
        (
            "arg_value without arg_key",
            "</think><tool_call>get_time<arg_value>1</arg_value></tool_call>",
            unexpected(
                0,
                "get_time",
                "<arg_key> or </tool_call>",
                "<arg_value>1</arg_value>",
            ),
        ),
        (
            "a key without its value",
            "</think><tool_call>f<arg_key>a</arg_key></tool_call>",
            unexpected(0, "f", "<arg_value>", ""),
        ),
        (
            "an unterminated key",
            "</think><tool_call>f<arg_key>a<arg_value>1</arg_value></tool_call>",
            unexpected(0, "f", "</arg_key>", "a<arg_value>1</arg_value>"),
        ),
        (
            "text between tags",
            "</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value>junk</tool_call>",
            unexpected(0, "f", "<arg_key> or </tool_call>", "junk"),
        ),
        (
            "no function name",
            "</think><tool_call><arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
            GlmXmlError::NoName {
                index: 0,
                // The first 40 characters of the body.
                found: "<arg_key>a</arg_key><arg_value>1</arg_va".to_owned(),
            },
        ),
        (
            "a key given twice",
            "</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>a</arg_key><arg_value>2</arg_value></tool_call>",
            GlmXmlError::DuplicateKey {
                index: 0,
                name: "f".to_owned(),
                key: "a".to_owned(),
            },
        ),
        (
            "an integer argument that is not JSON",
            "</think><tool_call>get_weather<arg_key>days</arg_key><arg_value>three</arg_value></tool_call>",
            GlmXmlError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            },
        ),
        (
            "the second call malformed",
            "</think><tool_call>get_time</tool_call><tool_call>get_time<arg_value>1</arg_value></tool_call>",
            unexpected(
                1,
                "get_time",
                "<arg_key> or </tool_call>",
                "<arg_value>1</arg_value>",
            ),
        ),
    ]
}

/// Feeds `pieces` one by one; the deltas' sum, checked against the message.
fn streamed(
    prompt: &str,
    format: ReasoningFormat,
    pieces: &[&str],
) -> Result<Message, GlmXmlError> {
    let mut p = ChatParser::with_tools(prompt, format, glm());
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

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm_fixtures_parse_to_expected_messages() {
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

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm_malformed_calls_are_named_errors() {
    let mut wrong = Vec::new();
    for (name, raw, want) in malformed() {
        match streamed(ON, ReasoningFormat::Deepseek, &[raw]) {
            Err(e) if e == want => {}
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
/// character at a time, in two at every character, in three at every pair
/// of tag boundaries (a call split across three stream pieces at its tags).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm_parse_does_not_depend_on_chunking() {
    let cases = fixtures()
        .into_iter()
        .map(|(name, prompt, format, raw, want)| (name, prompt, format, raw, Ok(want)))
        .chain(
            malformed()
                .into_iter()
                .map(|(name, raw, e)| (name, ON, ReasoningFormat::Deepseek, raw, Err(e))),
        );
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

/// The calls parsed from the fixtures, rendered back through the template as
/// an assistant turn, are the fixtures' markup byte for byte: the parser's
/// text/JSON rule is the template's `tojson if v is not string` read back.
/// And the template's generation prompt is [`ON`] with or without a
/// reasoning effort outside its list.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm_fixtures_are_the_templates_markup() {
    let t = ChatTemplate::parse(TEMPLATE).expect("template");
    let render = |vars: Value| {
        let Value::Object(vars) = vars else {
            panic!("vars")
        };
        t.render(&vars).expect("render")
    };
    // Argument keys in sorted order: the engine iterates an object's keys
    // sorted, the model writes them in its own order.
    for raw in [TWO_CALLS, NOTE] {
        let m = streamed(ON, ReasoningFormat::Deepseek, &[raw]).expect("parses");
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
        let out = render(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": m.content, "tool_calls": calls},
            ],
            "add_generation_prompt": false,
        }));
        let (_, after) = raw.split_once("</think>").expect("a closed span");
        let want = format!("<|user|>hi<|assistant|><think></think>{after}");
        assert!(out.ends_with(&want), "{out}\n  does not end with\n{want}");
    }
    for effort in [json!(null), json!("medium")] {
        let mut vars = Map::new();
        vars.insert(
            "messages".into(),
            json!([{"role": "user", "content": "hi"}]),
        );
        vars.insert("add_generation_prompt".into(), json!(true));
        if !effort.is_null() {
            vars.insert("reasoning_effort".into(), effort);
        }
        assert_eq!(t.render(&vars).expect("render"), ON);
    }
}

/// The server reads the markup from its template: GLM's for the GLM
/// template, DSML for V4.1's and Qwen3's (neither spells `<arg_key>`).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tool_format_follows_the_template() {
    assert_eq!(ToolFormat::of_template(TEMPLATE), ToolFormat::GlmXml);
    assert_eq!(
        ToolFormat::of_template(common::V41_TEMPLATE),
        ToolFormat::Dsml
    );
    assert_eq!(
        ToolFormat::of_template(include_str!("fixtures/qwen3-chat-template.jinja")),
        ToolFormat::Dsml
    );
}

// ---------------------------------------------------------------- the server

fn start(script: &str) -> std::net::SocketAddr {
    common::start_templated(Box::new(ScriptedEngine::new(4096, script)), TEMPLATE)
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

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm_chat_tool_calls_stream_and_non_stream() {
    let addr = start(ONE_CALL);
    let r = post(addr, "/v1/chat/completions", &tool_body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(
        c["message"]["reasoning_content"], "Need the weather.",
        "{v}"
    );
    assert_eq!(c["message"]["content"], "", "{v}");
    let calls: Vec<Value> = c["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(
        calls,
        [json!({"name": "get_weather", "arguments": r#"{"location":"Seoul","days":3}"#})]
    );
    let r = post(addr, "/v1/chat/completions", &tool_body(true));
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["reasoning_content"], "Need the weather.");
    assert_eq!(folded["content"], "");
    assert_eq!(folded["calls"], json!(calls), "streamed == non-streamed");
}

/// Markup that does not parse is the request's named error: a 500 without a
/// stream, an error event that ends a stream (no `[DONE]`), for a malformed
/// call and for one the generation leaves open.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_glm_malformed_call_is_an_api_error() {
    for (script, want) in [
        (
            "</think><tool_call>get_time<arg_value>1</arg_value></tool_call>",
            "tool-call markup: tool call 0 (get_time): expected <arg_key> or </tool_call>",
        ),
        (
            "</think>Calling.<tool_call>get_time<arg_key>x",
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
        assert_eq!(r.status, 200, "{}", r.body);
        let e = fold_stream(&r.body).expect_err("an error event");
        assert!(e.starts_with(want), "{e}");
    }
}
