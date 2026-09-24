//! Gate: the reasoning split and DSML tool calls of `/v1/chat/completions`.
//!
//! Fixtures are raw model output as the V4.1 template's own markup spells it
//! (`tests/fixtures/v41-chat-template.jinja`, the GGUF's `tokenizer.chat_template`):
//!
//! ```text
//! {{- '\n\n<' + dsml_token + 'tool_calls>\n' -}}
//! {{- '<' + dsml_token + 'invoke name="' + func['name'] + '">\n' -}}
//! {{- '<' + dsml_token + 'parameter name="' + key + '" string="true">' + val + '</' + dsml_token + 'parameter>\n' -}}
//! {{- '<' + dsml_token + 'parameter name="' + key + '" string="false">' + (val | tojson) + '</' + dsml_token + 'parameter>\n' -}}
//! {%- if not args -%} {{- '\n' -}} {%- endif -%}
//! {{- '</' + dsml_token + 'invoke>\n' -}}
//! {{- '</' + dsml_token + 'tool_calls>' -}}
//! {{- '<｜end▁of▁sentence｜>' -}}
//! ```
//!
//! and the generation prompt `'<｜Assistant｜>'` + `thinking_start_token` (thinking
//! on) or `thinking_end_token` (off). `hw_fixtures_are_the_templates_markup` renders
//! the parsed calls back through the template and gets the fixture text again.

mod common;

use common::{V41_TEMPLATE, post, start, start_with};
use serde_json::{Map, Value, json};
use serve::dsml::{ChatParser, Message, ToolCall};
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const ON: &str = "<｜User｜>hi<｜Assistant｜><think>";
const OFF: &str = "<｜User｜>hi<｜Assistant｜></think>";

const ANSWER: &str = "The user greets me.</think>Hello!";
const PLAIN: &str = "Hello! a</think>b";
const ONE_CALL: &str = "Need the weather.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n<｜DSML｜parameter name=\"location\" string=\"true\">Seoul</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
const TWO_CALLS: &str = "Checking both.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Seoul</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"get_time\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
const DSML_IN_REASONING: &str = "I could answer with </｜DSML｜tool_calls> or with\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_time\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> but no.</think>Done.";
const TRUNCATED: &str = "Still thinking about <｜DSML｜tool_calls>\n<｜DSML｜inv";
const QUOTED: &str = "</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"note\">\n<｜DSML｜parameter name=\"text\" string=\"true\">He said \"hi\" — 서울 ☃\n\\n</｜DSML｜parameter>\n<｜DSML｜parameter name=\"filter\" string=\"false\">{\"q\": \"a\\\"b\", \"n\": [1, 2]}</｜DSML｜parameter>\n<｜DSML｜parameter name=\"loose\" string=\"false\">not json</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
const OPEN_BLOCK: &str = "Calling.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_time\">\n";
const MALFORMED: &str =
    "x\n\n<｜DSML｜tool_calls>\n<｜DSML｜call get_time/>\n</｜DSML｜tool_calls> y";

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

/// `(name, prompt, format, tools, raw output, expected message)`.
fn fixtures() -> Vec<(
    &'static str,
    &'static str,
    ReasoningFormat,
    bool,
    &'static str,
    Message,
)> {
    use ReasoningFormat::{Deepseek, None};
    let weather = || call(0, "get_weather", r#"{"days":3,"location":"Seoul"}"#);
    vec![
        (
            "thinking on, answer",
            ON,
            Deepseek,
            false,
            ANSWER,
            msg("The user greets me.", "Hello!", vec![]),
        ),
        (
            "thinking off",
            OFF,
            Deepseek,
            false,
            PLAIN,
            msg("", PLAIN, vec![]),
        ),
        (
            "thinking on, format none",
            ON,
            None,
            false,
            ANSWER,
            msg("", ANSWER, vec![]),
        ),
        (
            "thinking on, one call",
            ON,
            Deepseek,
            true,
            ONE_CALL,
            msg("Need the weather.", "", vec![weather()]),
        ),
        (
            "thinking off, two calls",
            OFF,
            Deepseek,
            true,
            TWO_CALLS,
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
            "DSML inside reasoning",
            ON,
            Deepseek,
            true,
            DSML_IN_REASONING,
            msg(
                DSML_IN_REASONING
                    .strip_suffix("</think>Done.")
                    .expect("fixture"),
                "Done.",
                vec![],
            ),
        ),
        (
            "DSML inside reasoning, format none",
            ON,
            None,
            true,
            DSML_IN_REASONING,
            msg("", DSML_IN_REASONING, vec![]),
        ),
        (
            "truncated reasoning",
            ON,
            Deepseek,
            true,
            TRUNCATED,
            msg(TRUNCATED, "", vec![]),
        ),
        (
            "quotes and unicode",
            OFF,
            Deepseek,
            true,
            QUOTED,
            msg(
                "",
                "</think>",
                vec![call(
                    0,
                    "note",
                    r#"{"text":"He said \"hi\" — 서울 ☃\n\\n","filter":{"n":[1,2],"q":"a\"b"},"loose":"not json"}"#,
                )],
            ),
        ),
        (
            "block open at the end",
            OFF,
            Deepseek,
            true,
            OPEN_BLOCK,
            msg("", OPEN_BLOCK, vec![]),
        ),
        (
            "malformed block",
            OFF,
            Deepseek,
            true,
            MALFORMED,
            msg("", MALFORMED, vec![]),
        ),
        (
            "tools off",
            OFF,
            Deepseek,
            false,
            TWO_CALLS,
            msg("", TWO_CALLS, vec![]),
        ),
    ]
}

fn streamed(prompt: &str, format: ReasoningFormat, tools: bool, pieces: &[&str]) -> Message {
    let mut p = ChatParser::new(prompt, format, tools);
    let mut total = Message::default();
    for piece in pieces {
        let d = p.push(piece);
        total.reasoning.push_str(&d.reasoning);
        total.content.push_str(&d.content);
        total.calls.extend(d.calls);
    }
    let d = p.finish();
    total.reasoning.push_str(&d.reasoning);
    total.content.push_str(&d.content);
    total.calls.extend(d.calls);
    assert_eq!(&total, p.message(), "deltas sum to the message");
    total
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_fixtures_parse_to_expected_messages() {
    let mut wrong = Vec::new();
    for (name, prompt, format, tools, raw, want) in fixtures() {
        let got = ChatParser::parse(prompt, format, tools, raw);
        if got != want {
            wrong.push(format!("{name}:\n  got  {got:?}\n  want {want:?}"));
            continue;
        }
        if let Some(c) = got.calls.first() {
            let v: Value = serde_json::from_str(&c.arguments).expect("arguments are JSON");
            assert!(v.is_object(), "{name}: arguments are an object");
        }
        assert!(
            want.calls.is_empty() || !got.content.contains("｜DSML｜"),
            "{name}: parsed DSML must not reach content"
        );
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
fn hw_parse_does_not_depend_on_chunking() {
    for (name, prompt, format, tools, raw, want) in fixtures() {
        let chars: Vec<String> = raw.chars().map(String::from).collect();
        let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            streamed(prompt, format, tools, &pieces),
            want,
            "{name}: char by char"
        );
        for (at, _) in raw.char_indices().skip(1) {
            let (a, b) = raw.split_at(at);
            assert_eq!(
                streamed(prompt, format, tools, &[a, b]),
                want,
                "{name}: split at {at}"
            );
        }
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_fixtures_are_the_templates_markup() {
    let t = ChatTemplate::parse(V41_TEMPLATE).expect("template");
    let m = ChatParser::parse(OFF, ReasoningFormat::Deepseek, true, TWO_CALLS);
    let calls: Vec<Value> = m
        .calls
        .iter()
        .map(|c| json!({"id": "x", "type": "function", "function": {"name": c.name, "arguments": c.arguments}}))
        .collect();
    let mut vars = Map::new();
    vars.insert(
        "messages".into(),
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": m.content, "tool_calls": calls},
        ]),
    );
    vars.insert("bos_token".into(), json!(""));
    let rendered = t.render(&vars).expect("render");
    let want = format!("<｜User｜>hi<｜Assistant｜></think>{TWO_CALLS}<｜end▁of▁sentence｜>");
    assert_eq!(rendered, want);
    let mut vars = Map::new();
    vars.insert(
        "messages".into(),
        json!([{"role": "user", "content": "hi"}]),
    );
    vars.insert("bos_token".into(), json!(""));
    vars.insert("add_generation_prompt".into(), json!(true));
    assert_eq!(
        t.render(&vars).expect("render"),
        OFF,
        "thinking defaults to off"
    );
    vars.insert("enable_thinking".into(), json!(true));
    assert_eq!(t.render(&vars).expect("render"), ON);
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_format_values() {
    assert_eq!(
        ReasoningFormat::from_request(None),
        Ok(ReasoningFormat::Deepseek)
    );
    assert_eq!(
        ReasoningFormat::from_request(Some(&json!("auto"))),
        Ok(ReasoningFormat::Deepseek)
    );
    assert_eq!(
        ReasoningFormat::from_request(Some(&json!("none"))),
        Ok(ReasoningFormat::None)
    );
    assert!(ReasoningFormat::from_request(Some(&json!("deepseek-legacy"))).is_err());
}

// ---------------------------------------------------------------- the server

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
    ])
}

fn tool_body(extra: Value) -> Value {
    let mut b = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "weather and time?"}],
        "temperature": 0,
        "tools": tools(),
    });
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e);
    }
    b
}

/// A streamed chat body folded back into `(finish_reason, message)`: deltas
/// concatenated, one entry per tool-call index.
fn fold_stream(body: &str) -> (Value, Value) {
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut finish = Value::Null;
    for e in body.split("\n\n").filter_map(|e| e.strip_prefix("data: ")) {
        if e == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(e).expect("chunk JSON");
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
            assert!(!s.contains("｜DSML｜"), "DSML reached delta.content: {s}");
            content.push_str(s);
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            let i = usize::try_from(tc["index"].as_u64().expect("index")).expect("small");
            assert_eq!(i, calls.len(), "each call streams once, in order");
            let mut tc = tc.clone();
            tc.as_object_mut().expect("object").remove("index");
            calls.push(tc);
        }
    }
    let mut m = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        m["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        m["tool_calls"] = json!(calls);
    }
    (finish, m)
}

/// Replaces the per-request nonce inside tool-call ids.
fn call_ids(m: &mut Value) -> Vec<String> {
    let mut ids = Vec::new();
    for tc in m["tool_calls"].as_array_mut().into_iter().flatten() {
        let id = tc["id"].as_str().expect("id").to_owned();
        let (head, nonce) = id.rsplit_once('_').expect("call_<i>_<nonce>");
        assert!(head.starts_with("call_") && nonce.len() == 16, "{id}");
        tc["id"] = json!(head);
        ids.push(id);
    }
    ids
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_chat_tool_calls_stream_and_non_stream() {
    for (script, extra, reasoning) in [
        (TWO_CALLS, json!({}), ""),
        (
            ONE_CALL,
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            "Need the weather.",
        ),
    ] {
        let addr = start_with(Box::new(ScriptedEngine::new(4096, script)));
        let r = post(addr, "/v1/chat/completions", &tool_body(extra.clone()));
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let c = &v["choices"][0];
        assert_eq!(c["finish_reason"], "tool_calls", "{v}");
        let mut whole = c["message"].clone();
        let calls = whole["tool_calls"].as_array().expect("tool_calls").clone();
        assert!(!calls.is_empty());
        for tc in &calls {
            assert_eq!(tc["type"], "function");
            assert!(
                tc["function"]["arguments"].is_string(),
                "arguments is a JSON string: {tc}"
            );
            serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().expect("s"))
                .expect("arguments parse");
        }
        assert_eq!(
            whole
                .get("reasoning_content")
                .and_then(Value::as_str)
                .unwrap_or(""),
            reasoning
        );
        assert!(
            !whole["content"]
                .as_str()
                .expect("content")
                .contains("｜DSML｜")
        );
        let ids = call_ids(&mut whole);
        let mut unique = ids.clone();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "ids unique: {ids:?}");

        let mut sb = tool_body(extra);
        sb["stream"] = json!(true);
        let r = post(addr, "/v1/chat/completions", &sb);
        assert_eq!(r.status, 200, "{}", r.body);
        let (finish, mut folded) = fold_stream(&r.body);
        assert_eq!(finish, "tool_calls");
        let sids = call_ids(&mut folded);
        assert_ne!(sids, ids, "a second request gets its own nonce");
        assert_eq!(folded, whole, "streamed == non-streamed");
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_tool_choice_none_leaves_dsml_as_content() {
    let addr = start_with(Box::new(ScriptedEngine::new(4096, TWO_CALLS)));
    let v = post(
        addr,
        "/v1/chat/completions",
        &tool_body(json!({"tool_choice": "none"})),
    )
    .json();
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["message"]["content"], TWO_CALLS);
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
    let r = post(
        addr,
        "/v1/chat/completions",
        &tool_body(json!({"reasoning_format": "deepseek-legacy"})),
    );
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("reasoning_format"), "{}", r.body);
}

fn chat_body(extra: Value) -> Value {
    let mut b = json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "abc"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0,
    });
    if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
        b.extend(e);
    }
    b
}

/// The body with its per-request values (`id`, `created`, `timings`) replaced by `_`.
fn mask(body: &str) -> String {
    let mut s = body.to_owned();
    for (key, close) in [
        ("\"id\":\"", '"'),
        ("\"created\":", ','),
        ("\"timings\":{", '}'),
    ] {
        let mut from = 0;
        while let Some(at) = s[from..].find(key) {
            let start = from + at + key.len();
            let end = start + s[start..].find(close).expect("closed");
            s.replace_range(start..end, "_");
            from = start + 1;
        }
    }
    s
}

/// Recorded from the server before the split existed (base 6853f02), same request.
const GOLDEN: &str = r#"{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"abc","role":"assistant"}}],"created":_,"id":"_","model":"m","object":"chat.completion","timings":{_},"usage":{"completion_tokens":4,"prompt_tokens":15,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":19}}"#;
const GOLDEN_STREAM: &str = concat!(
    r#"data: {"choices":[{"delta":{"content":null,"role":"assistant"},"finish_reason":null,"index":0}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"delta":{"content":"a"},"finish_reason":null,"index":0}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"delta":{"content":"b"},"finish_reason":null,"index":0}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"delta":{"content":"c"},"finish_reason":null,"index":0}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[],"created":_,"id":"_","model":"m","object":"chat.completion.chunk","timings":{_},"usage":{"completion_tokens":4,"prompt_tokens":15,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":19}}"#,
    "\n\n",
    "data: [DONE]\n\n",
);

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_no_tools_format_none_is_byte_identical() {
    // Each request on a fresh slot, as the goldens were recorded: a second
    // request on the same slot keeps its cached prefix (`cached_tokens`).
    for format in [json!({"reasoning_format": "none"}), json!({})] {
        let r = post(
            start(4096),
            "/v1/chat/completions",
            &chat_body(format.clone()),
        );
        assert_eq!(mask(&r.body), GOLDEN, "{format}");
        let mut b = chat_body(format.clone());
        b["stream"] = json!(true);
        b["stream_options"] = json!({"include_usage": true});
        let r = post(start(4096), "/v1/chat/completions", &b);
        assert_eq!(mask(&r.body), GOLDEN_STREAM, "{format}");
    }
}
