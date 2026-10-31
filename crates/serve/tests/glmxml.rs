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
//!
//! The same scanner reads Solar-Open2's spelling of the grammar
//! ([`KvSpelling::Solar`]): `<|tool_call:start|>{name}\n<|tool_arg:start|>{key}
//! <|tool_arg:value|>{value}<|tool_arg:end|>\n…<|tool_call:end|>`, the template's
//! `render_tool_instruction` and `render_tool_arguments`. Its tests are the
//! `hw_solar_*` ones at the end; the fixtures there are that markup written out,
//! under no reasoning split (Solar's think tags are `reasoning.rs`'s).

mod common;

use common::post;
use serde_json::{Map, Value, json};
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, Spelling, ToolCall, ToolFormat, Tools};
use serve::glmxml::{ArgTypes, KvError, KvSpelling};
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

fn scan(spelling: KvSpelling) -> Option<Tools> {
    Some(Tools::KvMarkup(
        spelling,
        ArgTypes::of_tools(Some(&tools())),
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
fn malformed() -> Vec<(&'static str, &'static str, KvError)> {
    let unexpected = |index, name: &str, expected: &str, found: &str| KvError::Unexpected {
        index,
        name: name.to_owned(),
        expected: expected.to_owned(),
        found: found.to_owned(),
    };
    vec![
        (
            "unterminated",
            "</think><tool_call>get_time<arg_key>x</arg_key>",
            KvError::Unterminated {
                index: 0,
                close: "</tool_call>",
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
            "a key cut short by the next argument's tags",
            "</think><tool_call>f<arg_key>a<arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value></tool_call>",
            // The first 40 characters after the first `<arg_key>`.
            unexpected(
                0,
                "f",
                "</arg_key>",
                "a<arg_value>1</arg_value><arg_key>b</arg",
            ),
        ),
        (
            "text between tags",
            "</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value>junk</tool_call>",
            unexpected(0, "f", "<arg_key> or </tool_call>", "junk"),
        ),
        (
            "no function name",
            "</think><tool_call><arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
            KvError::NoName {
                index: 0,
                // The first 40 characters of the body.
                found: "<arg_key>a</arg_key><arg_value>1</arg_va".to_owned(),
            },
        ),
        (
            "a key given twice",
            "</think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>a</arg_key><arg_value>2</arg_value></tool_call>",
            KvError::DuplicateKey {
                index: 0,
                name: "f".to_owned(),
                key: "a".to_owned(),
            },
        ),
        (
            "an integer argument that is not JSON",
            "</think><tool_call>get_weather<arg_key>days</arg_key><arg_value>three</arg_value></tool_call>",
            KvError::NotJson {
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
) -> Result<Message, MarkupError> {
    streamed_as(KvSpelling::Glm, prompt, format, pieces)
}

/// [`streamed`] under `spelling`'s scan.
fn streamed_as(
    spelling: KvSpelling,
    prompt: &str,
    format: ReasoningFormat,
    pieces: &[&str],
) -> Result<Message, MarkupError> {
    let mut p = ChatParser::with_tools(prompt, format, scan(spelling));
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
    // The arguments keep the model's key order through the parse and the
    // template's `.items()`.
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

/// The server reads the markup from its template's source spellings: GLM's
/// for the GLM template, DSML for V4.1's; Qwen3's source spells `<tool_call>`
/// with neither family's other tags, so the source alone names no parser —
/// its rendered call does, which the hermes gate pins.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tool_format_follows_the_template() {
    assert_eq!(
        ToolFormat::of_template(TEMPLATE),
        ToolFormat::KvMarkup(KvSpelling::Glm)
    );
    assert_eq!(
        ToolFormat::of_template(common::V41_TEMPLATE),
        ToolFormat::Dsml(Spelling::Joined)
    );
    assert_eq!(
        ToolFormat::of_template(include_str!("fixtures/qwen3-chat-template.jinja")),
        ToolFormat::Unparsed
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
                !s.contains("<tool_call>") && !s.contains("<|tool_call:start|>"),
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
        let r = post(
            addr,
            "/v1/chat/completions",
            &common::unwatched(tool_body(false)),
        );
        assert_eq!(r.status, 500, "{}", r.body);
        let m = r.json()["error"]["message"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        assert!(m.starts_with(want), "{m}");
        let r = post(
            addr,
            "/v1/chat/completions",
            &common::unwatched(tool_body(true)),
        );
        assert_eq!(r.status, 200, "{}", r.body);
        let e = fold_stream(&r.body).expect_err("an error event");
        assert!(e.starts_with(want), "{e}");
    }
}

// ---------------------------------------------------------- Solar-Open's spelling

/// A synthetic chat template that teaches Solar-Open2's markup in its tool
/// header, the line the real template's `render_tool_instruction` writes: it
/// types [`KvSpelling::Solar`] by its source, as the real one does, and renders
/// a conversation and a generation prompt with no think span.
const SOLAR_TEMPLATE: &str = concat!(
    "{%- for message in messages %}",
    "{{- '<|im:start|>' + message.role + '<|im:content|>' + message.content + '<|im:end|>\\n' }}",
    "{%- endfor %}",
    "{%- if tools %}{{- 'Each tool call MUST use this following format: ",
    "<|tool_call:start|>{name}\\n<|tool_arg:start|>{key}<|tool_arg:value|>{value}",
    "<|tool_arg:end|>\\n<|tool_call:end|>\\n' }}{%- endif %}",
    "{%- if add_generation_prompt %}{{- '<|im:start|>assistant<|im:content|>' }}{%- endif %}",
);

/// What the template writes for `get_weather(location="Seoul", days=3)`.
const SOLAR_ONE: &str = "<|tool_call:start|>get_weather\n<|tool_arg:start|>location<|tool_arg:value|>Seoul<|tool_arg:end|>\n<|tool_arg:start|>days<|tool_arg:value|>3<|tool_arg:end|>\n<|tool_call:end|>\n";
/// Content, then two calls, a line feed after each: the second takes no arguments.
const SOLAR_TWO: &str = "Checking both.<|tool_call:start|>get_weather\n<|tool_arg:start|>location<|tool_arg:value|>Seoul<|tool_arg:end|>\n<|tool_call:end|>\n<|tool_call:start|>get_time\n<|tool_call:end|>\n";
/// A JSON value, and a string that opens with a space and holds a `<`, a
/// newline and the start of a tag.
const SOLAR_NOTE: &str = "<|tool_call:start|>save_note\n<|tool_arg:start|>filter<|tool_arg:value|>{\"n\": [1, 2], \"q\": \"서울 ☃\"}<|tool_arg:end|>\n<|tool_arg:start|>text<|tool_arg:value|> a<b & \"c\" </tool\nand <|tool<|tool_arg:end|>\n<|tool_call:end|>\n";
const SOLAR_TYPED: &str = "<|tool_call:start|>get_weather\n<|tool_arg:start|>location<|tool_arg:value|>123<|tool_arg:end|>\n<|tool_arg:start|>days<|tool_arg:value|>3<|tool_arg:end|>\n<|tool_call:end|>\n";
const SOLAR_UNTYPED: &str = "<|tool_call:start|>lookup\n<|tool_arg:start|>q<|tool_arg:value|>42<|tool_arg:end|>\n<|tool_arg:start|>w<|tool_arg:value|>word<|tool_arg:end|>\n<|tool_arg:start|>o<|tool_arg:value|>{\"a\": null}<|tool_arg:end|>\n<|tool_call:end|>\n";
const SOLAR_EMPTY: &str = "<|tool_call:start|>get_weather\n<|tool_arg:start|>location<|tool_arg:value|><|tool_arg:end|>\n<|tool_call:end|>";
/// No line feed anywhere: the scan skips whitespace, it does not need it.
const SOLAR_BARE: &str = "<|tool_call:start|>get_weather<|tool_arg:start|>location<|tool_arg:value|>Seoul<|tool_arg:end|><|tool_call:end|>";
/// After a call's close only the one line feed is the markup's.
const SOLAR_AFTER: &str = "<|tool_call:start|>get_time\n<|tool_call:end|>\n\nDone.";
const SOLAR_GLUED: &str = "<|tool_call:start|>get_time\n<|tool_call:end|> after";
const SOLAR_NEAR_TAG: &str = "a <|tool_ca b <|tool_call:start";

fn solar_streamed(pieces: &[&str]) -> Result<Message, MarkupError> {
    streamed_as(KvSpelling::Solar, "", ReasoningFormat::None, pieces)
}

/// `(name, raw output, expected message)`.
fn solar_fixtures() -> Vec<(&'static str, &'static str, Message)> {
    vec![
        (
            "the template's example, one call",
            SOLAR_ONE,
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul","days":3}"#)],
            ),
        ),
        (
            "two calls in one message, the line feeds the markup's",
            SOLAR_TWO,
            msg(
                "",
                "Checking both.",
                vec![
                    call(0, "get_weather", r#"{"location":"Seoul"}"#),
                    call(1, "get_time", "{}"),
                ],
            ),
        ),
        (
            "a JSON-valued argument, and a string with spaces, < and a newline kept whole",
            SOLAR_NOTE,
            msg(
                "",
                "",
                vec![call(
                    0,
                    "save_note",
                    r#"{"filter":{"n":[1,2],"q":"서울 ☃"},"text":" a<b & \"c\" </tool\nand <|tool"}"#,
                )],
            ),
        ),
        (
            "a string-typed value that reads as JSON stays text",
            SOLAR_TYPED,
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"123","days":3}"#)],
            ),
        ),
        (
            "an unknown function's values: JSON, else text",
            SOLAR_UNTYPED,
            msg(
                "",
                "",
                vec![call(0, "lookup", r#"{"q":42,"w":"word","o":{"a":null}}"#)],
            ),
        ),
        (
            "an empty string value",
            SOLAR_EMPTY,
            msg("", "", vec![call(0, "get_weather", r#"{"location":""}"#)]),
        ),
        (
            "no line feeds between tags",
            SOLAR_BARE,
            msg(
                "",
                "",
                vec![call(0, "get_weather", r#"{"location":"Seoul"}"#)],
            ),
        ),
        (
            "a second line feed after a call is content",
            SOLAR_AFTER,
            msg("", "\nDone.", vec![call(0, "get_time", "{}")]),
        ),
        (
            "text glued to a call's close is content",
            SOLAR_GLUED,
            msg("", " after", vec![call(0, "get_time", "{}")]),
        ),
        ("no call", "Hello!", msg("", "Hello!", vec![])),
        (
            "text that is not a call opening, and one cut short at the end",
            SOLAR_NEAR_TAG,
            msg("", SOLAR_NEAR_TAG, vec![]),
        ),
    ]
}

/// `(name, raw output, the error)`.
fn solar_malformed() -> Vec<(&'static str, &'static str, KvError)> {
    let unexpected = |index, name: &str, expected: &str, found: &str| KvError::Unexpected {
        index,
        name: name.to_owned(),
        expected: expected.to_owned(),
        found: found.to_owned(),
    };
    vec![
        (
            "unterminated",
            "<|tool_call:start|>get_time\n<|tool_arg:start|>x<|tool_arg:value|>",
            KvError::Unterminated {
                index: 0,
                close: "<|tool_call:end|>",
                text: "<|tool_call:start|>get_time\n<|tool_arg:start|>x<|tool_arg:value|>"
                    .to_owned(),
            },
        ),
        (
            "an argument end without its start",
            "<|tool_call:start|>get_time\n<|tool_arg:end|><|tool_call:end|>",
            unexpected(
                0,
                "get_time",
                "<|tool_arg:start|> or <|tool_call:end|>",
                "<|tool_arg:end|>",
            ),
        ),
        (
            "a key without its value, cut short by the next argument's tags",
            "<|tool_call:start|>f\n<|tool_arg:start|>a<|tool_arg:end|>\n<|tool_arg:start|>b<|tool_arg:value|>1<|tool_arg:end|>\n<|tool_call:end|>",
            // The first 40 characters after the first `<|tool_arg:start|>`.
            unexpected(
                0,
                "f",
                "<|tool_arg:value|>",
                "a<|tool_arg:end|>\n<|tool_arg:start|>b<|t",
            ),
        ),
        (
            "a value that never ends",
            "<|tool_call:start|>f\n<|tool_arg:start|>a<|tool_arg:value|>1<|tool_call:end|>",
            unexpected(0, "f", "<|tool_arg:end|>", "1"),
        ),
        (
            "text between tags",
            "<|tool_call:start|>f\n<|tool_arg:start|>a<|tool_arg:value|>1<|tool_arg:end|>\njunk<|tool_call:end|>",
            unexpected(0, "f", "<|tool_arg:start|> or <|tool_call:end|>", "junk"),
        ),
        (
            "no function name",
            "<|tool_call:start|><|tool_arg:start|>a<|tool_arg:value|>1<|tool_arg:end|><|tool_call:end|>",
            KvError::NoName {
                index: 0,
                // The first 40 characters of the body.
                found: "<|tool_arg:start|>a<|tool_arg:value|>1<|".to_owned(),
            },
        ),
        (
            "a line feed where the name belongs",
            "<|tool_call:start|>\n<|tool_call:end|>",
            KvError::NoName {
                index: 0,
                found: "\n".to_owned(),
            },
        ),
        (
            "a key given twice",
            "<|tool_call:start|>f\n<|tool_arg:start|>a<|tool_arg:value|>1<|tool_arg:end|>\n<|tool_arg:start|>a<|tool_arg:value|>2<|tool_arg:end|>\n<|tool_call:end|>",
            KvError::DuplicateKey {
                index: 0,
                name: "f".to_owned(),
                key: "a".to_owned(),
            },
        ),
        (
            "an integer argument that is not JSON",
            "<|tool_call:start|>get_weather\n<|tool_arg:start|>days<|tool_arg:value|>three<|tool_arg:end|>\n<|tool_call:end|>",
            KvError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            },
        ),
        (
            "the second call malformed",
            "<|tool_call:start|>get_time\n<|tool_call:end|>\n<|tool_call:start|>get_time\n<|tool_arg:end|><|tool_call:end|>",
            unexpected(
                1,
                "get_time",
                "<|tool_arg:start|> or <|tool_call:end|>",
                "<|tool_arg:end|>",
            ),
        ),
    ]
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_solar_fixtures_parse_to_expected_messages() {
    let mut wrong = Vec::new();
    for (name, raw, want) in solar_fixtures() {
        match solar_streamed(&[raw]) {
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
fn hw_solar_malformed_calls_are_named_errors() {
    let mut wrong = Vec::new();
    for (name, raw, want) in solar_malformed() {
        match solar_streamed(&[raw]) {
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

/// A message, or an error, never depends on how Solar's text was cut: the
/// cuts of [`hw_glm_parse_does_not_depend_on_chunking`], at every byte of the
/// line feeds and tags too.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_solar_parse_does_not_depend_on_chunking() {
    let cases = solar_fixtures()
        .into_iter()
        .map(|(name, raw, want)| (name, raw, Ok(want)))
        .chain(
            solar_malformed()
                .into_iter()
                .map(|(name, raw, e)| (name, raw, Err(MarkupError::from(e)))),
        );
    for (name, raw, want) in cases {
        for pieces in common::every_chunking(raw, 1) {
            assert_eq!(solar_streamed(&pieces), want, "{name}: {pieces:?}");
        }
    }
}

/// A streaming Solar parser over `pieces`: the message, and the call pieces
/// it reported; `finish` ends the stream.
fn solar_pieces(pieces: &[&str], finish: bool) -> (Message, Vec<CallPiece>) {
    let mut p = ChatParser::with_tools("", ReasoningFormat::None, scan(KvSpelling::Solar))
        .streaming_calls();
    let mut out = Vec::new();
    for piece in pieces {
        p.try_push(piece, &[]).expect("parses");
        out.extend(p.take_pieces());
    }
    if finish {
        p.try_finish().expect("closes");
        out.extend(p.take_pieces());
    }
    (p.message().clone(), out)
}

/// Under every cut a Solar call streams the arguments it closes with: the
/// pieces concatenate to the call the parser that does not stream reads, and
/// the text of a string-typed argument goes out as it is written.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_solar_every_cut_streams_the_arguments_the_call_closes_with() {
    for (name, raw, want) in solar_fixtures() {
        if want.calls.is_empty() {
            continue;
        }
        for pieces in common::every_chunking(raw, 3) {
            let (got, streamed) = solar_pieces(&pieces, true);
            assert_eq!(got, want, "{name}: {pieces:?}");
            let mut calls: Vec<(String, String)> = Vec::new();
            for piece in &streamed {
                match piece {
                    CallPiece::Open { index, name: call } => {
                        assert_eq!(*index, calls.len(), "{name}: calls open in order");
                        calls.push((call.clone(), String::new()));
                    }
                    CallPiece::Args { index, text } => calls[*index].1.push_str(text),
                }
            }
            let closed: Vec<(String, String)> = got
                .calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect();
            assert_eq!(calls, closed, "{name}: {pieces:?}");
        }
    }
    let at = SOLAR_NOTE.find("</tool").expect("in the text value");
    let (_, streamed) = solar_pieces(&[&SOLAR_NOTE[..at]], false);
    let sent: String = streamed
        .iter()
        .filter_map(|p| match p {
            CallPiece::Args { text, .. } => Some(text.as_str()),
            CallPiece::Open { .. } => None,
        })
        .collect();
    assert!(
        sent.ends_with(r#""text":" a<b & \"c\" "#),
        "the text so far: {sent}"
    );
}

/// A scan reads the one spelling it is given: each spelling's markup is
/// content to the other's scan.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_each_spelling_reads_only_its_own_markers() {
    const GLM_CALL: &str =
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Seoul</arg_value></tool_call>";
    for (spelling, raw) in [(KvSpelling::Glm, SOLAR_ONE), (KvSpelling::Solar, GLM_CALL)] {
        let got = streamed_as(spelling, "", ReasoningFormat::None, &[raw]).expect("no markup");
        assert_eq!(got, msg("", raw, vec![]), "{spelling:?}");
    }
}

/// The server reads Solar's spelling from its template's source, and GLM's
/// template stays GLM's.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_solar_tool_format_follows_the_template() {
    let solar = ToolFormat::KvMarkup(KvSpelling::Solar);
    assert_eq!(ToolFormat::of_template(SOLAR_TEMPLATE), solar);
    let parsed = ChatTemplate::parse(SOLAR_TEMPLATE).expect("template parses");
    assert_eq!(ToolFormat::of_chat_template(&parsed), solar);
    assert_ne!(ToolFormat::of_template(TEMPLATE), solar);
}

fn solar_server(script: &str) -> std::net::SocketAddr {
    common::start_templated(Box::new(ScriptedEngine::new(4096, script)), SOLAR_TEMPLATE)
}

/// A Solar call through `/v1/chat/completions`, streamed and not: the line
/// feeds around the calls are not content, and the streamed calls fold to the
/// non-streamed ones.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_solar_chat_tool_calls_stream_and_non_stream() {
    let addr = solar_server(SOLAR_TWO);
    let r = post(addr, "/v1/chat/completions", &tool_body(false));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
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
    let r = post(addr, "/v1/chat/completions", &tool_body(true));
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["content"], "Checking both.");
    assert_eq!(folded["calls"], json!(calls), "streamed == non-streamed");
}
