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
use serve::dsml::{ChatParser, MarkupError, Message, Spelling, ToolCall, ToolFormat, Tools};
use serve::glmxml::KvSpelling;
use serve::qwenxml::{Framing, ParamKinds, QwenXmlError};
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const TEMPLATE: &str = include_str!("fixtures/qwen38-chat-template.jinja");
const QWEN36: &str = include_str!("fixtures/qwen36-chat-template.jinja");
const QWEN3: &str = include_str!("fixtures/qwen3-chat-template.jinja");
/// MiMo-V2.6's template (the RL and MOPD checkpoints' are byte-identical): it
/// renders a call compactly, `<parameter=KEY>VALUE</parameter>`.
const MIMO: &str = include_str!("fixtures/mimo-chat-template.jinja");

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
/// A JSON-typed value whose strings hold a raw newline and a raw tab (the
/// spelling a model writing code reaches: strict JSON forbids the raw byte,
/// its meaning still the character itself), with escapes of its own that stay
/// as written.
const RAW_CONTROL: &str = "</think><tool_call>\n<function=save_note>\n<parameter=tags>\n[\"a \\\"quoted\\\" line\nnext\", \"tab\tback\\\\slash\"]\n</parameter>\n</function>\n</tool_call>";
const BOOLEANS: &str = "</think><tool_call>\n<function=save_note>\n<parameter=pinned>\nTrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=save_note>\n<parameter=pinned>\n FALSE \n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=save_note>\n<parameter=pinned>\nfalse\n</parameter>\n</function>\n</tool_call>";
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
/// A first call that opens bare, `<function=NAME>` with no `<tool_call>`
/// before it: an omission Qwen3-Coder makes now and then.
const BARE_CALL: &str =
    "<function=run_python>\n<parameter=code>\nls -la\n</parameter>\n</function>\n</tool_call>";

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

fn qwen(framing: Framing) -> Option<Tools> {
    Some(Tools::QwenXml(
        framing,
        ParamKinds::of_tools(Some(&tools())),
    ))
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
            "a JSON value whose strings hold raw control characters, escaped",
            ON,
            Deepseek,
            RAW_CONTROL,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "save_note",
                    r#"{"tags":["a \"quoted\" line\nnext", "tab\tback\\slash"]}"#,
                )],
            ),
        ),
        (
            "a boolean in either case, with whitespace around it, is spelled true or false",
            ON,
            Deepseek,
            BOOLEANS,
            msg(
                "",
                "",
                vec![
                    call(0, "save_note", r#"{"pinned":true}"#),
                    call(1, "save_note", r#"{"pinned":false}"#),
                    call(2, "save_note", r#"{"pinned":false}"#),
                ],
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
        (
            "a bare call is text under a template that teaches a think span",
            CLOSED,
            Deepseek,
            BARE_CALL,
            msg("", BARE_CALL, vec![]),
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
            "a boolean parameter that is neither true nor false",
            "</think><tool_call>\n<function=save_note>\n<parameter=pinned>\nyes\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::NotBoolean {
                index: 0,
                name: "save_note".to_owned(),
                key: "pinned".to_owned(),
                value: "yes".to_owned(),
            },
        ),
        (
            "an empty boolean value with no line of its own",
            "</think><tool_call>\n<function=save_note>\n<parameter=pinned>\n</parameter>\n</function>\n</tool_call>",
            QwenXmlError::Unterminated {
                index: 0,
                text: "<tool_call>\n<function=save_note>\n<parameter=pinned>\n</parameter>\n</function>\n</tool_call>".to_owned(),
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

/// Qwen3-Coder's generation prompt after one user turn: its template teaches
/// no think span, so the answer starts at once.
const CODER_ON: &str = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n";

/// `(name, raw output, the message or the error)` under [`Framing::Coder`],
/// after [`CODER_ON`]: a template that teaches no think span lets its model
/// open its first call bare, `<function=NAME>` naming a tool of the request,
/// the call still closed by `</tool_call>` (llama.cpp's `tool_call_first`).
/// A name the tools do not hold is no call, and a second call needs its own
/// `<tool_call>`.
fn bare_fixtures() -> Vec<(&'static str, &'static str, Result<Message, MarkupError>)> {
    vec![
        (
            "the first call opens bare, a string parameter framed as usual",
            BARE_CALL,
            Ok(msg(
                "",
                "",
                vec![call(0, "run_python", r#"{"code":"ls -la"}"#)],
            )),
        ),
        (
            "a bare call with two parameters, typed as usual",
            "<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
            Ok(msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            )),
        ),
        (
            "text, then the first call opens bare",
            "Checking.\n<function=get_time>\n</function>\n</tool_call>",
            Ok(msg("", "Checking.\n", vec![call(0, "get_time", "{}")])),
        ),
        (
            "a bare call whose name the tools do not hold stays text",
            "<function=get_date>\n</function>\n</tool_call>",
            Ok(msg(
                "",
                "<function=get_date>\n</function>\n</tool_call>",
                vec![],
            )),
        ),
        (
            "a second call that opens bare stays text",
            "<tool_call>\n<function=get_time>\n</function>\n</tool_call>\n<function=get_time>\n</function>\n</tool_call>",
            Ok(msg(
                "",
                "\n<function=get_time>\n</function>\n</tool_call>",
                vec![call(0, "get_time", "{}")],
            )),
        ),
        (
            "a bare call is still not closed at the end",
            "<function=get_time>\n</function>\n",
            Err(MarkupError::Qwen(QwenXmlError::Unterminated {
                index: 0,
                text: "<function=get_time>\n</function>\n".to_owned(),
            })),
        ),
        (
            "a bare call's parameters are typed as usual",
            "<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>",
            Err(MarkupError::Qwen(QwenXmlError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            })),
        ),
    ]
}

/// Feeds `pieces` one by one; the deltas' sum, checked against the message.
fn streamed(
    framing: Framing,
    prompt: &str,
    format: ReasoningFormat,
    pieces: &[&str],
) -> Result<Message, MarkupError> {
    let mut p = ChatParser::with_tools(prompt, format, qwen(framing));
    let mut total = Message::default();
    let mut add = |d: Message| {
        total.reasoning.push_str(&d.reasoning);
        total.content.push_str(&d.content);
        total.calls.extend(d.calls);
    };
    for piece in pieces {
        add(p.try_push(piece, &[])?);
    }
    add(p.try_finish()?);
    assert_eq!(&total, p.message(), "deltas sum to the message");
    Ok(total)
}

/// The fixtures parse to their messages, their arguments JSON objects.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen_fixtures_parse_to_expected_messages() {
    let mut wrong = Vec::new();
    for (name, prompt, format, raw, want) in fixtures() {
        match streamed(Framing::Lines, prompt, format, &[raw]) {
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
        match streamed(Framing::Lines, ON, ReasoningFormat::Deepseek, &[raw]) {
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
        for pieces in common::every_chunking(raw, 1) {
            assert_eq!(
                streamed(Framing::Lines, prompt, format, &pieces),
                want,
                "{name}: {pieces:?}"
            );
        }
    }
}

/// The first call a template that teaches no think span may open bare parses
/// to its message, its errors named, and no answer depends on how the text
/// was cut (the fixtures' own gate pattern).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_first_call_may_open_bare_under_qwen3coders_template() {
    for (name, raw, want) in bare_fixtures() {
        for pieces in common::every_chunking(raw, 1) {
            assert_eq!(
                streamed(Framing::Coder, CODER_ON, ReasoningFormat::Deepseek, &pieces),
                want,
                "{name}: {pieces:?}"
            );
        }
    }
}

// ------------------------------------------------------- the compact framing

/// The framing the server reads from the MiMo template's rendered call; any
/// other format is a named failure.
fn mimo_framing() -> Framing {
    let t = ChatTemplate::parse(MIMO).expect("template parses");
    match ToolFormat::of_chat_template(&t) {
        ToolFormat::QwenXml(framing) => framing,
        other => panic!("MiMo's template types to {other:?}, not Qwen3-Coder's markup"),
    }
}

/// The MiMo template's prompt after one user turn, thinking off.
fn mimo_prompt() -> String {
    let t = ChatTemplate::parse(MIMO).expect("template parses");
    let Value::Object(vars) = json!({
        "messages": [{"role": "user", "content": "q"}],
        "add_generation_prompt": true,
        "enable_thinking": false,
    }) else {
        panic!("vars")
    };
    t.render(&vars).expect("render")
}

/// `get_weather(city: string)`, the one tool of the llama.cpp reference's run.
fn weather_tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}}])
}

/// A template types its markup's framing from the call it renders, not from
/// its name or from its source's spelling alone: MiMo's compact call is
/// [`Framing::Compact`], the Qwen3.6, Qwen3.8 and Nemotron calls (a value on a
/// line of its own, a think span the render shows) are [`Framing::Lines`],
/// and so framed but with no think span — a template whose first call may
/// open bare ([`Framing::Coder`]) — are Qwen3-Coder's own and two small
/// templates that differ from it in nothing but how they write a value. The
/// probe's own call decides, not a header's example before it. A template
/// that spells the tags and renders no call is [`ToolFormat::Unparsed`]: no
/// framing is guessed.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_framing_follows_the_rendered_call() {
    let compact = ToolFormat::QwenXml(Framing::Compact);
    let lines = ToolFormat::QwenXml(Framing::Lines);
    let coder = ToolFormat::QwenXml(Framing::Coder);
    let t = ChatTemplate::parse(MIMO).expect("template parses");
    assert_eq!(ToolFormat::of_chat_template(&t), compact);
    assert_eq!(ToolFormat::of_template(MIMO), compact);
    for name in ["qwen36", "qwen38", "nemotron", "nemotron-bolt"] {
        let source = template_named(name);
        let t = ChatTemplate::parse(source).expect("template parses");
        assert_eq!(ToolFormat::of_chat_template(&t), lines, "{name}");
        assert_eq!(ToolFormat::of_template(source), lines, "{name}");
    }
    let small = |value: &str| {
        format!(
            "{{%- for m in messages -%}}{{%- if m.tool_calls -%}}{{%- for c in m.tool_calls -%}}\
             <tool_call><function={{{{ c.function.name }}}}>\
             {{%- for k, v in c.function.arguments.items() -%}}\
             <parameter={{{{ k }}}}>{value}</parameter>\
             {{%- endfor -%}}</function></tool_call>\
             {{%- endfor -%}}{{%- endif -%}}{{%- endfor -%}}"
        )
    };
    assert_eq!(ToolFormat::of_template(&small("{{ v }}")), compact);
    assert_eq!(
        ToolFormat::of_template(
            &small("\n{{ v }}\n").replace("</parameter>", "</parameter>{{ '\\n' }}")
        ),
        coder
    );
    let example = "<tool_call><function=get_weather><parameter=location>\nSeoul\n</parameter>\n</function></tool_call>";
    assert_eq!(
        ToolFormat::of_template(&format!("{example}{}", small("{{ v }}"))),
        compact
    );
    let no_call = "{{- raise_exception('nothing renders') }}<tool_call><function=f><parameter=p>";
    assert_eq!(ToolFormat::of_template(no_call), ToolFormat::Unparsed);
}

/// The two outputs the llama.cpp reference's autoparser (mainline,
/// `tools/ref/chat_parse_ref`, MiMo's template, thinking off, one
/// `get_weather(city: string)`) parses, to the arguments it reads: `peg-native`
/// format, `{"city":"Seoul"}` for a value bare between its tags, and
/// `{"city":"\nSeoul\n"}` for one the model wrote on lines of its own — under
/// the compact framing the newlines are the value's bytes. The same text under
/// every cut yields the same call.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_compact_calls_parse_as_the_reference_reads_them() {
    let framing = mimo_framing();
    assert_eq!(framing, Framing::Compact);
    let prompt = mimo_prompt();
    let cases = [
        (
            "a value bare between its tags",
            "<tool_call>\n<function=get_weather>\n<parameter=city>Seoul</parameter></function>\n</tool_call>",
            r#"{"city":"Seoul"}"#,
        ),
        (
            "a value on lines of its own",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>",
            r#"{"city":"\nSeoul\n"}"#,
        ),
    ];
    for (name, raw, arguments) in cases {
        let want = msg("", "", vec![call(0, "get_weather", arguments)]);
        for pieces in common::every_chunking(raw, 1) {
            let mut p = ChatParser::with_tools(
                &prompt,
                ReasoningFormat::Deepseek,
                Some(Tools::QwenXml(
                    framing,
                    ParamKinds::of_tools(Some(&weather_tools())),
                )),
            );
            for piece in &pieces {
                p.try_push(piece, &[])
                    .unwrap_or_else(|e| panic!("{name}: {e}: {pieces:?}"));
            }
            p.try_finish()
                .unwrap_or_else(|e| panic!("{name}: {e}: {pieces:?}"));
            assert_eq!(p.message(), &want, "{name}: {pieces:?}");
        }
    }
}

/// `(name, raw output, expected message)` under [`Framing::Compact`]: a value
/// is the bytes between `<parameter=KEY>` and `</parameter>`, and whitespace
/// between tags is skipped, as the reference's parser reads the markup. The
/// parameter's schema types the value as it does under the lines framing.
fn compact_fixtures() -> Vec<(&'static str, &'static str, Message)> {
    vec![
        (
            "the template's own call, no whitespace at all",
            "<tool_call><function=get_weather><parameter=location>Seoul</parameter><parameter=days>3</parameter></function></tool_call>",
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            ),
        ),
        (
            "newlines between the tags are skipped",
            "<tool_call>\n<function=get_weather>\n<parameter=location>Seoul</parameter>\n<parameter=days>3</parameter>\n</function>\n</tool_call>",
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            ),
        ),
        (
            "a value's own newlines are its bytes",
            "<tool_call><function=get_weather><parameter=location>\nSeoul\n</parameter></function></tool_call>",
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"\nSeoul\n"}"#)],
            ),
        ),
        (
            "the lines framing's value is a value with a newline before the close",
            "<tool_call><function=get_weather><parameter=location>Seoul\n</parameter></function></tool_call>",
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul\n"}"#)],
            ),
        ),
        (
            "indentation and trailing newlines are kept",
            "<tool_call><function=run_python><parameter=code>    print(\"hi\")\n\n</parameter></function></tool_call>",
            msg(
                "",
                "",
                vec![call(0, "run_python", r#"{"code":"    print(\"hi\")\n\n"}"#)],
            ),
        ),
        (
            "an empty value",
            "<tool_call><function=run_python><parameter=code></parameter></function></tool_call>",
            msg("", "", vec![call(0, "run_python", r#"{"code":""}"#)]),
        ),
        (
            "object, array and boolean values typed by the schema",
            "<tool_call><function=save_note><parameter=filter>{\"n\": [1, 2], \"q\": \"서울 ☃\"}</parameter><parameter=tags>[\"a\", \"b\"]</parameter><parameter=pinned>true</parameter><parameter=text>a<b & \"c\"</parameter></function></tool_call>",
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
            "near-tags in a value, and a function close inside it",
            "<tool_call><function=run_python><parameter=code>a </parameter x> </tool_ca x> </function></parameter></function></tool_call>",
            msg(
                "",
                "",
                vec![call(
                    0,
                    "run_python",
                    r#"{"code":"a </parameter x> </tool_ca x> </function>"}"#,
                )],
            ),
        ),
        (
            "text before the first call, two calls",
            "Let me check.<tool_call><function=get_time></function></tool_call><tool_call><function=get_weather><parameter=location>Busan</parameter></function></tool_call>",
            msg(
                "",
                "Let me check.",
                vec![
                    call(0, "get_time", "{}"),
                    call(1, "get_weather", r#"{"location":"Busan"}"#),
                ],
            ),
        ),
    ]
}

/// Markup that does not parse under [`Framing::Compact`] is still a named
/// error: the schema's typing, the tags and the unclosed value.
fn compact_malformed() -> Vec<(&'static str, &'static str, QwenXmlError)> {
    vec![
        (
            "an unclosed value at the end",
            "<tool_call><function=get_weather><parameter=location>Seoul",
            QwenXmlError::Unterminated {
                index: 0,
                text: "<tool_call><function=get_weather><parameter=location>Seoul".to_owned(),
            },
        ),
        (
            "an integer parameter that is not JSON",
            "<tool_call><function=get_weather><parameter=days>three</parameter></function></tool_call>",
            QwenXmlError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            },
        ),
        (
            "a parameter the schema lacks",
            "<tool_call><function=get_weather><parameter=unit>C</parameter></function></tool_call>",
            QwenXmlError::UnknownParameter {
                index: 0,
                name: "get_weather".to_owned(),
                key: "unit".to_owned(),
            },
        ),
        (
            "a parameter given twice",
            "<tool_call><function=get_weather><parameter=location>Seoul</parameter><parameter=location>Busan</parameter></function></tool_call>",
            QwenXmlError::DuplicateParameter {
                index: 0,
                name: "get_weather".to_owned(),
                key: "location".to_owned(),
            },
        ),
        (
            "text between the parameters",
            "<tool_call><function=get_time>junk</function></tool_call>",
            QwenXmlError::Unexpected {
                index: 0,
                name: "get_time".to_owned(),
                expected: "a parameter or the function's close",
                found: "junk</function></tool_call>".to_owned(),
            },
        ),
        (
            "the function's close not followed by the call's",
            "<tool_call><function=get_time></function>done",
            QwenXmlError::Unexpected {
                index: 0,
                name: "get_time".to_owned(),
                expected: "</tool_call>",
                found: "done".to_owned(),
            },
        ),
    ]
}

/// The compact tables parse to their messages and errors, whole and under
/// every cut, in the framing the MiMo template's call types to.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_compact_framing_parses_and_names_its_errors_under_every_cut() {
    let framing = mimo_framing();
    assert_eq!(framing, Framing::Compact);
    let prompt = mimo_prompt();
    let cases = compact_fixtures()
        .into_iter()
        .map(|(name, raw, want)| (name, raw, Ok(want)))
        .chain(
            compact_malformed()
                .into_iter()
                .map(|(name, raw, e)| (name, raw, Err(MarkupError::from(e)))),
        );
    for (name, raw, want) in cases {
        for pieces in common::every_chunking(raw, 1) {
            assert_eq!(
                streamed(framing, &prompt, ReasoningFormat::Deepseek, &pieces),
                want,
                "{name}: {pieces:?}"
            );
        }
    }
}

/// The calls the MiMo template writes, parsed and rendered back through the
/// template, are the template's markup byte for byte.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_compact_calls_are_the_mimo_templates_markup() {
    let t = ChatTemplate::parse(MIMO).expect("template");
    let render = |calls: Value| {
        let Value::Object(vars) = json!({
            "messages": [
                {"role": "user", "content": "Note it."},
                {"role": "assistant", "content": "", "tool_calls": calls},
            ],
            "tools": [],
            "add_generation_prompt": false,
        }) else {
            panic!("vars")
        };
        t.render(&vars).expect("render")
    };
    let first = render(json!([
        {"type": "function", "function": {"name": "save_note", "arguments": {
            "filter": {"n": [1, 2], "q": "서울 ☃"}, "text": "a<b & \"c\""}}},
        {"type": "function", "function": {"name": "run_python", "arguments": {
            "code": "def f():\n    return 1\n"}}},
    ]));
    // The prompt through the assistant's closed span, the script the calls.
    let (head, script) = first
        .split_once("<think></think>")
        .expect("the assistant's closed span");
    let prompt = format!("{head}<think></think>");
    let script = script
        .strip_suffix("<|im_end|>")
        .expect("the assistant turn's end");
    let mut p = ChatParser::with_tools(&prompt, ReasoningFormat::Deepseek, qwen(mimo_framing()));
    let _ = p.try_push(script, &[]).expect("the markup parses");
    let _ = p.try_finish().expect("the markup finishes");
    let m = p.message().clone();
    assert_eq!(m.content, "", "{m:?}");
    assert_eq!(
        m.calls,
        vec![
            call(
                0,
                "save_note",
                r#"{"filter":{"n": [1, 2], "q": "서울 ☃"},"text":"a<b & \"c\""}"#,
            ),
            call(1, "run_python", r#"{"code":"def f():\n    return 1\n"}"#),
        ],
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
    assert_eq!(render(json!(calls)), first);
}

/// Every generative seat's markup, the parser the server reads from it
/// ([`ToolFormat::of_chat_template`]) by [`FIXTURE_TEMPLATES`] name: a seat
/// whose markup has no parser refuses every chat request that carries tools.
fn seat_format(name: &str) -> ToolFormat {
    match name {
        "v41" => ToolFormat::Dsml(Spelling::Joined),
        "glm5" => ToolFormat::KvMarkup(KvSpelling::Glm),
        "qwen3" => ToolFormat::Hermes,
        "qwen36" | "qwen38" | "nemotron" | "nemotron-bolt" => ToolFormat::QwenXml(Framing::Lines),
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
    assert_eq!(
        ToolFormat::of_template(TEMPLATE),
        ToolFormat::QwenXml(Framing::Lines)
    );
    assert_eq!(
        ToolFormat::of_template(QWEN36),
        ToolFormat::QwenXml(Framing::Lines)
    );
    assert_eq!(ToolFormat::of_template(QWEN3), ToolFormat::Unparsed);
    assert_eq!(
        ToolFormat::of_template(
            "<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value><function=F><parameter=P>"
        ),
        ToolFormat::KvMarkup(KvSpelling::Glm)
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
    let mut p = ChatParser::with_tools(&prompt, ReasoningFormat::Deepseek, qwen(Framing::Lines));
    let _ = p.try_push(script, &[]).expect("the markup parses");
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
            common::fold_call_delta(&mut calls, tc);
        }
    }
    let mut m = json!({"content": content, "reasoning_content": reasoning});
    m["calls"] = json!(
        calls
            .iter()
            .map(|c| c["function"].clone())
            .collect::<Vec<_>>()
    );
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

/// A call ends the span for the budget and loop guard without `</think>`,
/// on the chat route and on both text-completion routes, whose prompt ends
/// in `<think>`. The mock spells `<tool_call>` a byte an id: after `r` and
/// the first ten opener ids the budget of 12 has one left. Four-row passes
/// resume as soon as the opener ends, and a repeated argument stays whole.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_call_ends_the_span_for_the_budget_and_the_loop_guard() {
    use std::sync::{Arc, Mutex};

    use serve::{Drafted, Engine, EngineError, Tokenizer};

    struct DraftScript {
        inner: ScriptedEngine,
        produced: usize,
        passes: Arc<Mutex<Vec<usize>>>,
    }

    impl Engine for DraftScript {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            let id = self.inner.next(last, out)?;
            self.produced += 1;
            Ok(id)
        }
        fn advance(&mut self, mut last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
            self.passes.lock().expect("passes").push(self.produced);
            out.clear();
            for _ in 0..4 {
                last = self.next(last, None)?;
                out.push(last);
                if last == self.tokenizer().eos() {
                    break;
                }
            }
            Ok(Drafted {
                proposed: out.len() - 1,
                accepted: out.len() - 1,
            })
        }
        fn advance_rows(&self) -> usize {
            4
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.produced = 0;
            self.inner.reset()
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    let text = "a".repeat(300);
    let script = format!(
        "r<tool_call>\n<function=save_note>\n<parameter=text>\n{text}\n</parameter>\n</function>\n</tool_call>"
    );
    for drafting in [false, true] {
        let passes = Arc::new(Mutex::new(Vec::new()));
        let inner = ScriptedEngine::new(4096, &script);
        let engine: Box<dyn Engine> = if drafting {
            Box::new(DraftScript {
                inner,
                produced: 0,
                passes: Arc::clone(&passes),
            })
        } else {
            Box::new(inner)
        };
        let addr = start_templated(engine, TEMPLATE);
        for extra in [
            json!({"reasoning_budget": 12}),
            json!({"reasoning_loop": true}),
            json!({"reasoning_loop": true, "reasoning_budget": 12}),
        ] {
            let with = |mut body: Value| {
                passes.lock().expect("passes").clear();
                body.as_object_mut()
                    .expect("body")
                    .extend(extra.as_object().expect("extra").clone());
                body
            };
            let resumed = || {
                if drafting && extra.get("reasoning_budget").is_some() {
                    let passes = passes.lock().expect("passes");
                    assert!(
                        passes.contains(&12),
                        "a four-row pass must resume at the call opener: {extra}, {passes:?}"
                    );
                }
            };
            let mut chat = tool_body(false);
            chat["max_tokens"] = json!(1024);
            let r = post(addr, "/v1/chat/completions", &with(chat));
            assert_eq!(r.status, 200, "{drafting}, {extra}: {}", r.body);
            let v = r.json();
            let c = &v["choices"][0];
            assert_eq!(c["finish_reason"], "tool_calls", "{drafting}, {extra}: {v}");
            assert_eq!(c["message"]["reasoning_content"], "r", "{extra}: {v}");
            let args: Value = serde_json::from_str(
                c["message"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .expect("arguments"),
            )
            .expect("JSON");
            assert_eq!(args, json!({ "text": text }), "{extra}");
            resumed();
            let completion = json!({"prompt": "q<think>", "n_predict": 1024, "temperature": 0});
            let r = post(addr, "/completion", &with(completion));
            assert_eq!(r.status, 200, "{drafting}, {extra}: {}", r.body);
            let v = r.json();
            assert_eq!(v["stop_type"], "eos", "{drafting}, {extra}: {v}");
            assert_eq!(v["content"], script, "{drafting}, {extra}");
            resumed();
            let completion = json!({"prompt": "q<think>", "max_tokens": 1024, "temperature": 0});
            let r = post(addr, "/v1/completions", &with(completion));
            assert_eq!(r.status, 200, "{drafting}, {extra}: {}", r.body);
            let v = r.json();
            let c = &v["choices"][0];
            assert_eq!(c["finish_reason"], "stop", "{drafting}, {extra}: {v}");
            assert_eq!(c["text"], script, "{drafting}, {extra}");
            resumed();
        }
    }
}

/// A forced call with thinking on reasons first: the script's reasoning, with
/// a `<tool_call>` among it (the mock spells it a byte an id), runs to the
/// script's `</think>`, and only then is the named call written, on the chat
/// route whole and streamed and on `/completion`, whose prompt ends in
/// `<think>`. A script that ends inside the span is an error naming the
/// missing call.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_forced_call_follows_the_models_own_close() {
    const REASONING: &str = "r <tool_call> s";
    let named = json!({"type": "function", "function": {"name": "get_time"}});
    let called = json!([{"name": "get_time", "arguments": "{}"}]);
    let served = json!([tools()[0], tools()[1]]);
    let addr = start(&format!("{REASONING}</think>"));
    let mut body = tool_body(false);
    body["tools"] = served.clone();
    body["tool_choice"] = named.clone();
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["reasoning_content"], REASONING, "{v}");
    let calls: Vec<Value> = c["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(json!(calls), called, "{v}");
    body["stream"] = json!(true);
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["reasoning_content"], REASONING);
    assert_eq!(folded["calls"], called);
    let completion = json!({"prompt": "q<think>", "n_predict": 256, "temperature": 0,
        "tools": served, "tool_choice": named});
    let r = post(addr, "/completion", &completion);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(
        r.json()["content"],
        format!("{REASONING}</think>{CALL_IN_SPAN}"),
        "{}",
        r.body
    );
    let addr = start(REASONING);
    body["stream"] = json!(false);
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 500, "{}", r.body);
    assert!(r.body.contains("before the forced tool call"), "{}", r.body);
}

/// MiMo's compact calls return the reference arguments,
/// `get_weather {"city":"Seoul"}`, in streamed and non-streamed responses.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_server_on_mimos_template_answers_a_compact_call() {
    let script = "<tool_call>\n<function=get_weather>\n<parameter=city>Seoul</parameter></function>\n</tool_call>";
    let addr = start_templated(Box::new(ScriptedEngine::new(4096, script)), MIMO);
    let body = |stream: bool| {
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Seoul?"}],
            "temperature": 0,
            "tools": weather_tools(),
            "stream": stream,
        })
    };
    let want = json!({"name": "get_weather", "arguments": r#"{"city":"Seoul"}"#});
    let r = post(addr, "/v1/chat/completions", &body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["content"], "", "{v}");
    assert_eq!(c["message"]["tool_calls"][0]["function"], want, "{v}");
    let r = post(addr, "/v1/chat/completions", &body(true));
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["content"], "");
    assert_eq!(folded["calls"], json!([want]), "streamed == non-streamed");
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

/// A generation the token limit cuts inside a call is an answer, as llama-server answers it (its chat parser reads
/// the end of the text as the end of the input): 200 with `finish_reason` `length`. Cut before the function's name
/// is whole, the text read is `content` and no call is made; cut after it, the call stands with the arguments read
/// (`{`, then `{"location":"Seo` with a string value open), the text before it stays content, and a stream folds to
/// the same message. `/v1/messages` holds only closed calls: the cut call is text and `stop_reason` is
/// `max_tokens`. A generation that ends on its own inside a call stays the error
/// (`hw_qwen_malformed_call_is_an_api_error`).
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen_call_cut_by_the_limit_is_an_answer() {
    use serve::Tokenizer;
    const SCRIPT: &str = "</think>Calling.<tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n</function>\n</tool_call>";
    /// The cut: the script's start, the content a chat answer holds, the call it holds.
    type Cut = (
        &'static str,
        &'static str,
        Option<(&'static str, &'static str)>,
    );
    let cuts: [Cut; 3] = [
        (
            "</think>Calling.<tool_call>\n<func",
            "Calling.<tool_call>\n<func",
            None,
        ),
        (
            "</think>Calling.<tool_call>\n<function=get_weather>\n<para",
            "Calling.",
            Some(("get_weather", "{")),
        ),
        (
            "</think>Calling.<tool_call>\n<function=get_weather>\n<parameter=location>\nSeo",
            "Calling.",
            Some(("get_weather", r#"{"location":"Seo"#)),
        ),
    ];
    let addr = start(SCRIPT);
    for (read, content, call) in cuts {
        assert!(SCRIPT.starts_with(read), "{read:?} is the script's start");
        let tokens = serve::MockTokenizer.encode(read).len();
        let want: Vec<Value> = call
            .iter()
            .map(|(name, arguments)| json!({"name": name, "arguments": arguments}))
            .collect();
        let mut body = tool_body(false);
        body["max_tokens"] = json!(tokens);
        let r = post(addr, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{read:?}: {}", r.body);
        let v = r.json();
        let c = &v["choices"][0];
        assert_eq!(c["finish_reason"], "length", "{read:?}: {v}");
        assert_eq!(c["message"]["content"], content, "{read:?}: {v}");
        let calls: Vec<Value> = c["message"]["tool_calls"]
            .as_array()
            .map(|a| a.iter().map(|tc| tc["function"].clone()).collect())
            .unwrap_or_default();
        assert_eq!(calls, want, "{read:?}: {v}");

        body["stream"] = json!(true);
        let r = post(addr, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{read:?}: {}", r.body);
        let events: Vec<Value> = r
            .body
            .split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .filter(|e| *e != "[DONE]")
            .map(|e| serde_json::from_str(e).expect("chunk JSON"))
            .collect();
        assert!(
            events.iter().all(|e| e.get("error").is_none()),
            "{read:?}: {}",
            r.body
        );
        let finish = events
            .iter()
            .find_map(|e| e["choices"][0]["finish_reason"].as_str());
        assert_eq!(finish, Some("length"), "{read:?}: {}", r.body);
        let streamed: String = events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed, content, "{read:?}: streamed == non-streamed");
        if call.is_some() {
            let (_, folded) = fold_stream(&r.body).unwrap_or_else(|e| panic!("{read:?}: {e}"));
            assert_eq!(
                folded["calls"],
                json!(want),
                "{read:?}: streamed == non-streamed"
            );
        }

        let b = json!({
            "model": "m",
            "max_tokens": tokens,
            "messages": [{"role": "user", "content": "weather?"}],
            "temperature": 0,
            "tools": [{"name": "get_weather", "description": "",
                "input_schema": {"type": "object", "properties": {
                    "location": {"type": "string"}, "days": {"type": "integer"}}}}],
        });
        let r = post(addr, "/v1/messages", &b);
        assert_eq!(r.status, 200, "{read:?}: {}", r.body);
        let v = r.json();
        assert_eq!(v["stop_reason"], "max_tokens", "{read:?}: {v}");
        let blocks = v["content"].as_array().expect("content blocks");
        assert!(blocks.iter().all(|b| b["type"] == "text"), "{read:?}: {v}");
        let text: String = blocks.iter().filter_map(|b| b["text"].as_str()).collect();
        assert_eq!(
            Some(text.as_str()),
            read.strip_prefix("</think>"),
            "{read:?}: the text read, whole"
        );
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

// ---------------------------------------------------------------- Qwen3-Coder's own template

/// Qwen3-Coder-30B-A3B-Instruct's template, the checkpoint's `chat_template.jinja` byte for byte
/// (`tests/fixtures/qwen3coder-chat-template.source`). Its `tools` header filters a schema's keys with
/// `reject("in", ...)`, and it has no thinking span.
const CODER: &str = include_str!("fixtures/qwen3coder-chat-template.jinja");

/// `body` with each tool call's `arguments` that is a JSON string parsed to the object it encodes: the body a template
/// that walks `arguments` as an object renders, as the server hands it to one.
fn coder_arguments_parsed(body: &Value) -> Value {
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

/// Qwen3-Coder's chat template renders as the Hugging Face reference does: every case of the fixture through
/// `/apply-template` equals its reference render byte for byte. The case jinja2 refuses (tool-call `arguments` given as
/// a JSON string, which the template walks with `|items`) renders as the same body with the string parsed does, as the
/// server hands such a template the object. The cases reach `reject("in", ...)` over a schema's keys, a macro called
/// with a default argument, `|replace` chains on a key, `|tojson|safe` on a mapping, `|string` on a list and a boolean,
/// `loop.previtem` and `loop.nextitem` around a tool block, and the default identity line.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen3coder_template_renders_as_the_reference() {
    let addr = start_templated(Box::new(serve::MockEngine::new(4096)), CODER);
    let fixture: Value = serde_json::from_str(include_str!("fixtures/qwen3coder-renders.json"))
        .expect("fixture JSON");
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 13, "{fixture}");
    for case in cases {
        let r = post(addr, "/apply-template", &case["body"]);
        if case["error"].is_string() {
            assert_eq!(r.status, 200, "{}: {}", case["name"], r.body);
            let parsed = post(
                addr,
                "/apply-template",
                &coder_arguments_parsed(&case["body"]),
            );
            assert_eq!(parsed.status, 200, "{}: {}", case["name"], parsed.body);
            assert_eq!(
                r.json()["prompt"],
                parsed.json()["prompt"],
                "{}",
                case["name"]
            );
            assert!(
                r.json()["prompt"]
                    .as_str()
                    .is_some_and(|p| p.contains("<function=get_time>")),
                "{}: {}",
                case["name"],
                r.body
            );
            continue;
        }
        assert_eq!(r.status, 200, "{}: {}", case["name"], r.body);
        assert_eq!(r.json()["prompt"], case["prompt"], "{}", case["name"]);
    }
}

/// What picks the parser for Qwen3-Coder: the template's own spelling, `<tool_call>`, `<function=` and `<parameter=`,
/// and the framing read from the call its probe render writes, a value on a line of its own and no think span in the
/// render — the template whose first call may open bare ([`Framing::Coder`]). The probe renders the template with a
/// `tools` header, so a template the engine cannot render is [`ToolFormat::Unparsed`].
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_qwen3coder_template_is_read_as_qwen_markup() {
    let coder = ToolFormat::QwenXml(Framing::Coder);
    let t = ChatTemplate::parse(CODER).expect("template parses");
    assert_eq!(ToolFormat::of_chat_template(&t), coder);
    assert_eq!(ToolFormat::of_template(CODER), coder);
}

/// A server on Qwen3-Coder's template answers a request with tools: a completion in the model's markup becomes
/// OpenAI `tool_calls`, streamed and not, each parameter typed by the tool's schema (the string as written, a boolean
/// as `true` or `false` in either case, the others as the JSON the model wrote); the next request, which carries the call and its result (the call's `arguments` the
/// JSON string OpenAI sends), renders through the same template and is answered as text.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_server_on_qwen3coders_template_answers_a_call_and_its_result() {
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather", "description": "Current weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}},
        {"type": "function", "function": {"name": "set_options", "parameters": {"type": "object",
            "properties": {"verbose": {"type": "boolean"}, "retries": {"type": "integer"},
                           "ratio": {"type": "number"}, "tags": {"type": "array"},
                           "opts": {"type": "object"}, "note": {"type": "string"}}}}},
    ]);
    let script = "I will check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=set_options>\n<parameter=verbose>\nTrue\n</parameter>\n<parameter=retries>\n3\n</parameter>\n<parameter=ratio>\n0.5\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=opts>\n{\"k\": 1}\n</parameter>\n<parameter=note>\nfirst\nsecond\n</parameter>\n</function>\n</tool_call>";
    let addr = start_templated(Box::new(ScriptedEngine::new(4096, script)), CODER);
    let body = |stream: bool| {
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "What's the weather in Seoul?"}],
            "temperature": 0,
            "tools": tools,
            "stream": stream,
        })
    };
    let want = [
        json!({"name": "get_weather", "arguments": r#"{"city":"Seoul"}"#}),
        json!({"name": "set_options", "arguments":
            r#"{"verbose":true,"retries":3,"ratio":0.5,"tags":["a", "b"],"opts":{"k": 1},"note":"first\nsecond"}"#}),
    ];
    let r = post(addr, "/v1/chat/completions", &body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["content"], "I will check.\n", "{v}");
    let calls: Vec<Value> = c["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(calls, want, "{v}");
    let r = post(addr, "/v1/chat/completions", &body(true));
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["content"], "I will check.\n");
    assert_eq!(folded["calls"], json!(want), "streamed == non-streamed");

    let addr = start_templated(
        Box::new(ScriptedEngine::new(4096, "It is 21 degrees in Seoul.")),
        CODER,
    );
    let r = post(
        addr,
        "/v1/chat/completions",
        &json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "What's the weather in Seoul?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_a", "type": "function",
                    "function": {"name": "get_weather", "arguments": r#"{"city":"Seoul"}"#}}]},
                {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 21}"},
            ],
            "temperature": 0,
            "tools": tools,
        }),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "stop", "{v}");
    assert_eq!(c["message"]["content"], "It is 21 degrees in Seoul.", "{v}");
    assert!(c["message"]["tool_calls"].is_null(), "{v}");
}

/// A server on Qwen3-Coder's template answers a first call that opens bare,
/// `<function=NAME>` with no `<tool_call>` before it, streamed and not; a bare opener whose name the request's tools do not hold
/// stays text.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_server_on_qwen3coders_template_answers_a_bare_first_call() {
    let script = "I will check.\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>";
    let addr = start_templated(Box::new(ScriptedEngine::new(4096, script)), CODER);
    let body = |stream: bool| {
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "What's the weather in Seoul?"}],
            "temperature": 0,
            "tools": weather_tools(),
            "stream": stream,
        })
    };
    let want = json!({"name": "get_weather", "arguments": r#"{"city":"Seoul"}"#});
    let r = post(addr, "/v1/chat/completions", &body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["content"], "I will check.\n", "{v}");
    assert_eq!(c["message"]["tool_calls"][0]["function"], want, "{v}");
    let r = post(addr, "/v1/chat/completions", &body(true));
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["content"], "I will check.\n");
    assert_eq!(folded["calls"], json!([want]), "streamed == non-streamed");

    let unknown = "<function=get_date>\n</function>\n</tool_call>";
    let addr = start_templated(Box::new(ScriptedEngine::new(4096, unknown)), CODER);
    let r = post(addr, "/v1/chat/completions", &body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "stop", "{v}");
    assert_eq!(c["message"]["content"], unknown, "{v}");
    assert!(c["message"]["tool_calls"].is_null(), "{v}");
}
