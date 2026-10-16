//! Gate: Qwen3-Coder tool calls on the Qwen3.8 and Qwen3.6 chat templates
//! (`tests/fixtures/qwen38-chat-template.jinja`, the GGUF's
//! `tokenizer.chat_template`, and `qwen36-chat-template.jinja`, the HF repo's
//! `chat_template.jinja` byte for byte). Their `tools` header and assistant
//! `tool_calls` branch teach:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! and their thinking-on generation prompt ends `<|im_start|>assistant\n`
//! followed by `<think>\n`: the model's output starts inside the span, so a
//! call arrives after a `</think>` the model writes. The cases port
//! llama.cpp's Qwen3.5 tester rows (`special_function`, the parallel calls,
//! the multi-line and trailing-newline `code` values, the empty-argument
//! call, text before the first call) and its server rows (`Qwen3-Coder` with
//! a required tool) as the mock engine can run them.

mod common;

use common::{post, start_templated};
use serde_json::{Value, json};
use serve::dsml::{ChatParser, MarkupError, Message, ToolCall, ToolFormat, Tools};
use serve::qwenxml::{ParamKinds, QwenXmlError};
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const TEMPLATE: &str = include_str!("fixtures/qwen38-chat-template.jinja");
const QWEN36: &str = include_str!("fixtures/qwen36-chat-template.jinja");
const QWEN3: &str = include_str!("fixtures/qwen3-chat-template.jinja");

#[path = "common/fixture_templates.rs"]
mod fixture_templates;
use fixture_templates::FIXTURE_TEMPLATES;

/// The fixture template named `name`; a name [`FIXTURE_TEMPLATES`] lacks
/// fails by name.
fn template_named(name: &str) -> &'static str {
    FIXTURE_TEMPLATES
        .iter()
        .find(|&&(n, _)| n == name)
        .map_or_else(|| panic!("{name} is not a fixture template"), |&(_, s)| s)
}

/// The template's generation prompt after one user turn, thinking on: the
/// span is open, the script's first text already reasoning.
const ON: &str = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n";
/// A prompt the template closed (thinking off writes the empty span).
const CLOSED: &str =
    "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

const ONE_CALL: &str = "Need the weather.</think><tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
const TWO_CALLS: &str = "</think><tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
const CODE: &str = "</think><tool_call>\n<function=run_python>\n<parameter=code>\ndef hello():\n    print(\"Hello, world!\")\n\nhello()\n</parameter>\n</function>\n</tool_call>";
const TRAILING_NL: &str = "</think><tool_call>\n<function=run_python>\n<parameter=code>\nDone.\n\n</parameter>\n</function>\n</tool_call>";
const INDENT: &str = "</think><tool_call>\n<function=run_python>\n<parameter=code>\n    print(\"Hello, world!\")\n</parameter>\n</function>\n</tool_call>";
const TEXT_THEN_CALL: &str = "</think>Let me inspect the directory.\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
const THINK_THEN_CALL: &str = "I should inspect the directory.</think>Let me inspect it now.\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
const TYPED: &str = "</think><tool_call>\n<function=save_note>\n<parameter=filter>\n{\"n\": [1, 2], \"q\": \"서울 ☃\"}\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=pinned>\ntrue\n</parameter>\n<parameter=text>\na<b & \"c\"\n</parameter>\n</function>\n</tool_call>";
const LIST_AND_UNTYPED: &str = "</think><tool_call>\n<function=lookup>\n<parameter=q>\nhello, world\n</parameter>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>";
const RESOLVED: &str = "</think><tool_call>\n<function=resolve>\n<parameter=path>\nsrc/a.rs\n</parameter>\n<parameter=mode>\nfast\n</parameter>\n<parameter=when>\n2026-10-06\n</parameter>\n<parameter=either>\n7\n</parameter>\n<parameter=both>\nx\n</parameter>\n<parameter=limit>\n7\n</parameter>\n</function>\n</tool_call>";
const NO_PROPS: &str =
    "</think><tool_call>\n<function=empty_args_no_props>\n</function>\n</tool_call>";
const CALL_IN_SPAN: &str = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
const THOUGHT_THEN_CALL_IN_SPAN: &str =
    "Need to inspect the directory.\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
const CALL_THEN_TEXT: &str =
    "</think><tool_call>\n<function=get_time>\n</function>\n</tool_call> done.";
const FORMAT_NONE: &str =
    "Think.</think><tool_call>\n<function=get_time>\n</function>\n</tool_call>";
const AFTER_CLOSED: &str = "<tool_call>\n<function=get_time>\n</function>\n</tool_call> after";
const ANSWER: &str = "Hi.</think>Hello!";
const NEAR_TAG: &str = "</think>a <tool_ca b <tool_call";

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object",
            "properties": {}}}},
        {"type": "function", "function": {"name": "run_python", "parameters": {"type": "object",
            "properties": {"code": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "save_note", "parameters": {"type": "object",
            "properties": {"filter": {"type": "object"}, "tags": {"type": "array"},
                           "pinned": {"type": "boolean"}, "text": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object",
            "properties": {"q": {"type": ["string", "null"]}, "n": {}}}}},
        {"type": "function", "function": {"name": "empty_args_no_props",
            "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "resolve", "parameters": {"type": "object",
            "$defs": {"path": {"type": "string"}, "count": {"type": "integer"}},
            "properties": {"path": {"$ref": "#/$defs/path"}, "mode": {"enum": ["fast", 1]},
                           "when": {"format": "date"},
                           "either": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                           "both": {"allOf": [{"type": "string"}, {"maxLength": 4}]},
                           "limit": {"$ref": "#/$defs/count"}}}}},
    ])
}

fn qwen() -> Option<Tools> {
    Some(Tools::QwenXml(ParamKinds::of_tools(Some(&tools()))))
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
            "one call, a string and an integer parameter",
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
            "two calls, the second with no arguments",
            ON,
            Deepseek,
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
            "a multi-line string value, byte for byte",
            ON,
            Deepseek,
            CODE,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "run_python",
                    r#"{"code":"def hello():\n    print(\"Hello, world!\")\n\nhello()"}"#,
                )],
            ),
        ),
        (
            "a value that ends in a newline keeps it",
            ON,
            Deepseek,
            TRAILING_NL,
            msg("", "", vec![call(0, "run_python", r#"{"code":"Done.\n"}"#)]),
        ),
        (
            "a value that starts with indentation keeps it",
            ON,
            Deepseek,
            INDENT,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "run_python",
                    r#"{"code":"    print(\"Hello, world!\")"}"#,
                )],
            ),
        ),
        (
            "text before the call",
            ON,
            Deepseek,
            TEXT_THEN_CALL,
            msg(
                "",
                "Let me inspect the directory.\n",
                vec![call(0, "get_time", "{}")],
            ),
        ),
        (
            "a think span, text, then the call",
            ON,
            Deepseek,
            THINK_THEN_CALL,
            msg(
                "I should inspect the directory.",
                "Let me inspect it now.\n",
                vec![call(0, "get_time", "{}")],
            ),
        ),
        (
            "object, array and boolean values typed by the schema",
            ON,
            Deepseek,
            TYPED,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "save_note",
                    r#"{"filter":{"n": [1, 2], "q": "서울 ☃"},"tags":["a", "b"],"pinned":true,"text":"a<b & \"c\""}"#,
                )],
            ),
        ),
        (
            "a type list naming string is text, an untyped parameter JSON",
            ON,
            Deepseek,
            LIST_AND_UNTYPED,
            msg(
                "",
                "",
                vec![call(0, "lookup", r#"{"q":"hello, world","n":42}"#)],
            ),
        ),
        (
            "a schema that resolves to a string by $ref, enum, format, anyOf or allOf is text",
            ON,
            Deepseek,
            RESOLVED,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "resolve",
                    r#"{"path":"src/a.rs","mode":"fast","when":"2026-10-06","either":"7","both":"x","limit":7}"#,
                )],
            ),
        ),
        (
            "no args tool with no properties defined",
            ON,
            Deepseek,
            NO_PROPS,
            msg("", "", vec![call(0, "empty_args_no_props", "{}")]),
        ),
        (
            "a tool call ends the prefilled thinking block",
            ON,
            Deepseek,
            CALL_IN_SPAN,
            msg("", "", vec![call(0, "get_time", "{}")]),
        ),
        (
            "a tool call ends the thinking block after the model has thought",
            ON,
            Deepseek,
            THOUGHT_THEN_CALL_IN_SPAN,
            msg(
                "Need to inspect the directory.\n",
                "",
                vec![call(0, "get_time", "{}")],
            ),
        ),
        (
            "text after a call stays content",
            ON,
            Deepseek,
            CALL_THEN_TEXT,
            msg("", " done.", vec![call(0, "get_time", "{}")]),
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
fn malformed() -> Vec<(&'static str, &'static str, QwenXmlError)> {
    vec![
        (
            "an unclosed parameter at the end",
            "</think><tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul",
            QwenXmlError::Unterminated {
                index: 0,
                text: "<tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul".to_owned(),
            },
        ),
        (
            "a function opener with no name",
            "</think><tool_call>\n<function=>\n</function>\n</tool_call>",
            QwenXmlError::NoName {
                index: 0,
                found: "<function=>".to_owned(),
            },
        ),
        (
            "no function opener",
            "</think><tool_call>\nget_time\n</tool_call>",
            QwenXmlError::NoFunction {
                index: 0,
                found: "get_time\n</tool_call>".to_owned(),
            },
        ),
        (
            "an integer parameter that is not JSON",
            "</think><tool_call>\n<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            },
        ),
        (
            "an untyped parameter that is not JSON",
            "</think><tool_call>\n<function=lookup>\n<parameter=n>\nforty two\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::NotJson {
                index: 0,
                name: "lookup".to_owned(),
                key: "n".to_owned(),
                ty: "none".to_owned(),
                value: "forty two".to_owned(),
            },
        ),
        (
            "a function the tools do not name",
            "</think><tool_call>\n<function=get_date>\n</function>\n</tool_call>",
            QwenXmlError::UnknownFunction {
                index: 0,
                name: "get_date".to_owned(),
            },
        ),
        (
            "a parameter the schema lacks",
            "</think><tool_call>\n<function=get_weather>\n<parameter=unit>\nC\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::UnknownParameter {
                index: 0,
                name: "get_weather".to_owned(),
                key: "unit".to_owned(),
            },
        ),
        (
            "a parameter given twice",
            "</think><tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n<parameter=location>\nBusan\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::DuplicateParameter {
                index: 0,
                name: "get_weather".to_owned(),
                key: "location".to_owned(),
            },
        ),
        (
            "text between the parameters",
            "</think><tool_call>\n<function=get_time>\njunk</function>\n</tool_call>",
            QwenXmlError::Unexpected {
                index: 0,
                name: "get_time".to_owned(),
                expected: "a parameter or the function's close",
                found: "junk</function>\n</tool_call>".to_owned(),
            },
        ),
        (
            "the function's close not followed by the call's",
            "</think><tool_call>\n<function=get_time>\n</function>done",
            QwenXmlError::Unexpected {
                index: 0,
                name: "get_time".to_owned(),
                expected: "</tool_call>",
                found: "done".to_owned(),
            },
        ),
        (
            "no newline opening a value",
            "</think><tool_call>\n<function=get_weather>\n<parameter=location>Seoul\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::Unexpected {
                index: 0,
                name: "get_weather".to_owned(),
                expected: "a newline after <parameter=KEY>",
                found: "<parameter=location>Seoul\n</parameter>\n<".to_owned(),
            },
        ),
        (
            "the second call malformed",
            "</think><tool_call>\n<function=get_time>\n</function>\n</tool_call>\n<tool_call>\n<function=>\n</function>\n</tool_call>",
            QwenXmlError::NoName {
                index: 1,
                found: "<function=>".to_owned(),
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
    let mut p = ChatParser::with_tools(prompt, format, qwen());
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
fn hw_qwen_fixtures_parse_to_expected_messages() {
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
fn hw_qwen_malformed_calls_are_named_errors() {
    let mut wrong = Vec::new();
    for (name, raw, want) in malformed() {
        match streamed(ON, ReasoningFormat::Deepseek, &[raw]) {
            Err(e) if e == MarkupError::Qwen(want.clone()) => {}
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
fn hw_qwen_parse_does_not_depend_on_chunking() {
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

/// Every generative seat's markup, the parser the server reads from it
/// ([`ToolFormat::of_chat_template`]) by [`FIXTURE_TEMPLATES`] name: a seat
/// whose markup has no parser refuses every chat request that carries tools.
fn seat_format(name: &str) -> ToolFormat {
    match name {
        "v41" => ToolFormat::Dsml,
        "glm5" => ToolFormat::GlmXml,
        "qwen3" => ToolFormat::Hermes,
        "qwen36" | "qwen38" | "nemotron" | "nemotron-bolt" => ToolFormat::QwenXml,
        other => panic!("{other} is not in the seat table"),
    }
}

/// Each seat's template types to its parser, none to
/// [`ToolFormat::Unparsed`].
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_every_seat_template_has_a_parser() {
    let wrong: Vec<String> = FIXTURE_TEMPLATES
        .iter()
        .filter_map(|&(name, source)| {
            let t = ChatTemplate::parse(source).expect("template parses");
            let got = ToolFormat::of_chat_template(&t);
            let want = seat_format(name);
            (got != want).then(|| format!("{name}: {got:?}, want {want:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "seat(s) wrong:\n{}", wrong.join("\n"));
}

/// The source spelling types Qwen's markup as llama.cpp's rule does: a
/// source that spells `<tool_call>`, `<function=` and `<parameter=` (the
/// Qwen3.8 and Qwen3.6 templates), not Qwen3's, which spells `<tool_call>`
/// alone and is left to the rendered probe. GLM's tags come first: a source
/// that spells both families' tags is GLM's.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_detection_follows_the_spelling() {
    assert_eq!(ToolFormat::of_template(TEMPLATE), ToolFormat::QwenXml);
    assert_eq!(ToolFormat::of_template(QWEN36), ToolFormat::QwenXml);
    assert_eq!(ToolFormat::of_template(QWEN3), ToolFormat::Unparsed);
    assert_eq!(
        ToolFormat::of_template(
            "<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value><function=F><parameter=P>"
        ),
        ToolFormat::GlmXml
    );
}

/// The calls the template writes, parsed and rendered back through the
/// template, are the template's markup byte for byte: the parser's typing is
/// the template's `string if args_value is string else tojson` read back, and
/// a JSON parameter's value keeps the model's spelling.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen_fixtures_are_the_templates_markup() {
    let t = ChatTemplate::parse(TEMPLATE).expect("template");
    let render = |vars: Value| {
        let Value::Object(vars) = vars else {
            panic!("vars")
        };
        t.render(&vars).expect("render")
    };
    let tools = json!([{"type": "function", "function": {"name": "save_note", "parameters": {"type": "object",
        "properties": {"filter": {"type": "object"}, "text": {"type": "string"}}}}}]);
    let body = |calls: Value| {
        json!({
            "messages": [
                {"role": "user", "content": "Note it."},
                {"role": "assistant", "content": "", "tool_calls": calls},
            ],
            "tools": tools,
            "add_generation_prompt": false,
        })
    };
    let first = render(body(json!([{"type": "function", "function": {
        "name": "save_note",
        "arguments": {"filter": {"n": [1, 2], "q": "서울 ☃"}, "text": "a<b & \"c\""},
    }}])));
    // The prompt through the assistant's closed span, the script the calls.
    let (head, script) = first
        .split_once("</think>\n\n")
        .expect("the assistant's closed span");
    let prompt = format!("{head}</think>\n\n");
    // The turn's end is the stop token's text, not the generation's.
    let script = script
        .strip_suffix("<|im_end|>\n")
        .expect("the assistant turn's end");
    let mut p = ChatParser::with_tools(&prompt, ReasoningFormat::Deepseek, qwen());
    let _ = p.try_push(script).expect("the markup parses");
    let _ = p.try_finish().expect("the markup finishes");
    let m = p.message().clone();
    assert_eq!(m.content, "", "{m:?}");
    assert_eq!(
        m.calls,
        vec![call(
            0,
            "save_note",
            r#"{"filter":{"n": [1, 2], "q": "서울 ☃"},"text":"a<b & \"c\""}"#,
        )],
        "{m:?}"
    );
    let calls: Vec<Value> = m
        .calls
        .iter()
        .map(|c| {
            let args: Value = serde_json::from_str(&c.arguments).expect("JSON");
            json!({"type": "function", "function": {"name": c.name, "arguments": args}})
        })
        .collect();
    assert_eq!(render(body(json!(calls))), first);
}

// ---------------------------------------------------------------- the server

fn start(script: &str) -> std::net::SocketAddr {
    start_templated(Box::new(ScriptedEngine::new(4096, script)), TEMPLATE)
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
                !s.contains("<tool_call>") && !s.contains("<parameter="),
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

/// One call with its reasoning, streamed and not: `tool_calls` in OpenAI's
/// shape with `finish_reason` `tool_calls`, and the streamed deltas
/// concatenate to the non-streamed message.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen_chat_tool_calls_stream_and_non_stream() {
    let addr = start(ONE_CALL);
    let r = post(addr, "/v1/chat/completions", &tool_body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    let m = &c["message"];
    assert_eq!(m["reasoning_content"], "Need the weather.", "{v}");
    assert_eq!(m["content"], "", "{v}");
    let calls: Vec<Value> = m["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(
        calls,
        [json!({"name": "get_weather", "arguments": r#"{"location":"Seoul","days":3}"#})]
    );
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
    assert_eq!(folded["reasoning_content"], "Need the weather.");
    assert_eq!(folded["content"], "", "streamed == non-streamed");
    assert_eq!(folded["calls"], json!(calls), "streamed == non-streamed");
}

/// `/v1/messages` answers a generation with `tool_use` blocks and
/// `stop_reason` `tool_use`, the parameters typed by each tool's
/// `input_schema`, the text before them a `text` block.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen_messages_tool_use() {
    let addr = start(
        "</think>Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
    );
    let b = json!({
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "weather?"}],
        "temperature": 0,
        "tools": [{"name": "get_weather", "description": "",
            "input_schema": {"type": "object", "properties": {
                "location": {"type": "string"}, "days": {"type": "integer"}}}}],
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
    assert_eq!(
        last["input"],
        json!({"location": "Seoul", "days": 3}),
        "{v}"
    );
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
fn hw_qwen_malformed_call_is_an_api_error() {
    for (script, want) in [
        (
            "</think><tool_call>\n<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>",
            "tool-call markup: tool call 0 (get_weather): parameter days is typed integer, \
             and its value is not JSON",
        ),
        (
            "</think>Calling.<tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul",
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

/// The content after the think span, on every fixture template
/// ([`FIXTURE_TEMPLATES`]), as llama.cpp mainline's chat layer parses the same
/// generation (`tools/ref/chat_parse_ref` over each template, one user turn,
/// `reasoning_format` auto): the Qwen3-Coder parser (Qwen3.6, Qwen3.8,
/// Nemotron, Bolt) drops the whitespace that opens the content, with the span
/// open, closed by the prompt, or ended by a call; a span the model opens
/// itself (Qwen3's) drops the whitespace after its `</think>`; GLM's and
/// V4.1's keep it, as does Qwen3's prompt-closed span. Whitespace before a
/// call stays in the content on every template.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_content_after_the_span_is_mainlines() {
    let call = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
    // (name, thinking, with_tools, out, content, reasoning, calls)
    type Case = (
        &'static str,
        bool,
        bool,
        String,
        &'static str,
        Option<&'static str>,
        usize,
    );
    let cases: Vec<Case> = vec![
        (
            "qwen38",
            true,
            false,
            "r</think>\n\nHello".into(),
            "Hello",
            Some("r"),
            0,
        ),
        ("qwen38", false, false, "\nHello".into(), "Hello", None, 0),
        (
            "qwen36",
            true,
            false,
            "r</think>\n\nHello".into(),
            "Hello",
            Some("r"),
            0,
        ),
        ("qwen36", false, false, "\nHello".into(), "Hello", None, 0),
        (
            "nemotron",
            true,
            false,
            "r</think>\n\nHello".into(),
            "Hello",
            Some("r"),
            0,
        ),
        ("nemotron", false, false, "\nHello".into(), "Hello", None, 0),
        (
            "nemotron-bolt",
            true,
            false,
            "r</think>\n\nHello".into(),
            "Hello",
            Some("r"),
            0,
        ),
        (
            "nemotron-bolt",
            false,
            false,
            "\nHello".into(),
            "Hello",
            None,
            0,
        ),
        (
            "glm5",
            true,
            false,
            "r</think>\n\nHello".into(),
            "\n\nHello",
            Some("r"),
            0,
        ),
        (
            "glm5",
            false,
            false,
            "\nHello".into(),
            "",
            Some("\nHello"),
            0,
        ),
        (
            "v41",
            true,
            false,
            "r</think>\n\nHello".into(),
            "\n\nHello",
            Some("r"),
            0,
        ),
        ("v41", false, false, "\nHello".into(), "\nHello", None, 0),
        (
            "qwen3",
            true,
            false,
            "r</think>\n\nHello".into(),
            "r</think>\n\nHello",
            None,
            0,
        ),
        ("qwen3", false, false, "\nHello".into(), "\nHello", None, 0),
        (
            "qwen3",
            true,
            false,
            "<think>r</think>\n\nHello".into(),
            "Hello",
            Some("r"),
            0,
        ),
        (
            "qwen38",
            true,
            true,
            format!("r</think>\n\n{call}"),
            "",
            Some("r"),
            1,
        ),
        ("qwen38", false, true, format!("\n{call}\n"), "", None, 1),
        (
            "qwen38",
            false,
            true,
            format!("Let me check.\n{call}"),
            "Let me check.\n",
            None,
            1,
        ),
        (
            "qwen36",
            true,
            true,
            format!("r</think>\n\n{call}"),
            "",
            Some("r"),
            1,
        ),
        ("qwen36", false, true, format!("\n{call}\n"), "", None, 1),
        (
            "qwen36",
            false,
            true,
            format!("Let me check.\n{call}"),
            "Let me check.\n",
            None,
            1,
        ),
        (
            "nemotron",
            true,
            true,
            format!("r</think>\n\n{call}"),
            "",
            Some("r"),
            1,
        ),
        ("nemotron", false, true, format!("\n{call}\n"), "", None, 1),
        (
            "nemotron",
            false,
            true,
            format!("Let me check.\n{call}"),
            "Let me check.\n",
            None,
            1,
        ),
        (
            "nemotron-bolt",
            true,
            true,
            format!("r</think>\n\n{call}"),
            "",
            Some("r"),
            1,
        ),
        (
            "nemotron-bolt",
            false,
            true,
            format!("\n{call}\n"),
            "",
            None,
            1,
        ),
        (
            "nemotron-bolt",
            false,
            true,
            format!("Let me check.\n{call}"),
            "Let me check.\n",
            None,
            1,
        ),
        // C's isspace: the ideographic space stays, the vertical tab and form feed go.
        (
            "nemotron",
            false,
            false,
            "\u{3000}Hello".into(),
            "\u{3000}Hello",
            None,
            0,
        ),
        (
            "nemotron",
            false,
            false,
            "\x0B\x0CHello".into(),
            "Hello",
            None,
            0,
        ),
        (
            "qwen3",
            true,
            false,
            "<think>r</think>\u{3000}Hello".into(),
            "\u{3000}Hello",
            Some("r"),
            0,
        ),
    ];
    let no_case: Vec<&str> = FIXTURE_TEMPLATES
        .iter()
        .map(|&(name, _)| name)
        .filter(|name| !cases.iter().any(|c| c.0 == *name))
        .collect();
    assert!(
        no_case.is_empty(),
        "fixture template(s) with no case: {no_case:?}"
    );
    let mut wrong = Vec::new();
    for (name, thinking, with_tools, out, content, reasoning, calls) in cases {
        let template = template_named(name);
        let addr = start_templated(Box::new(ScriptedEngine::new(4096, &out)), template);
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}});
        if with_tools {
            body["tools"] = json!([{"type": "function", "function": {"name": "get_time",
                "description": "time", "parameters": {"type": "object", "properties": {}}}}]);
        }
        for stream in [false, true] {
            body["stream"] = json!(stream);
            let r = post(addr, "/v1/chat/completions", &body);
            assert_eq!(r.status, 200, "{name} {out:?}: {}", r.body);
            let (got_content, got_reasoning, got_calls) = if stream {
                let (_, folded) = fold_stream(&r.body).expect("no error event");
                let n = folded["calls"].as_array().map_or(0, Vec::len);
                (
                    folded["content"].clone(),
                    folded["reasoning_content"].clone(),
                    n,
                )
            } else {
                let v = r.json();
                let m = &v["choices"][0]["message"];
                let n = m["tool_calls"].as_array().map_or(0, Vec::len);
                (m["content"].clone(), m["reasoning_content"].clone(), n)
            };
            // The folded stream holds an absent `reasoning_content` as "".
            let want_reasoning = match (reasoning, stream) {
                (Some(r), _) => json!(r),
                (None, true) => json!(""),
                (None, false) => Value::Null,
            };
            if got_content != json!(content)
                || got_reasoning != want_reasoning
                || got_calls != calls
            {
                wrong.push(format!(
                    "{name} thinking={thinking} tools={with_tools} stream={stream} {out:?}: \
                     content {got_content} reasoning {got_reasoning} calls {got_calls}, \
                     want {content:?} {want_reasoning} {calls}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) differ from mainline:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// With `reasoning_format: none` nothing is split out and nothing is dropped:
/// the whitespace after a model-opened span's `</think>` stays in the content,
/// as llama.cpp's parser keeps it (`chat_parse_ref --reasoning-format none`:
/// `<think>r</think>\n\nHello` on Qwen3's template, with or without tools).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_format_none_drops_nothing() {
    let addr = start_templated(
        Box::new(ScriptedEngine::new(4096, "<think>r</think>\n\nHello")),
        QWEN3,
    );
    for with_tools in [false, true] {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0, "reasoning_format": "none"});
        if with_tools {
            body["tools"] = json!([{"type": "function", "function": {"name": "get_time",
                "description": "time", "parameters": {"type": "object", "properties": {}}}}]);
        }
        let r = post(addr, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            content.ends_with("</think>\n\nHello"),
            "tools={with_tools}: {v}"
        );
    }
}
