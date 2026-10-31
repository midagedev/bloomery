//! Gate: the reasoning split and DSML tool calls of `/v1/chat/completions`.
//!
//! Fixtures are raw model output as the V4 template's own markup spells it
//! (`tests/fixtures/v41-chat-template.jinja`, the GGUF's `tokenizer.chat_template`);
//! each runs a second time in V4.1's spelling, a space after the DSML token
//! (`tests/fixtures/v41-dsml-spaced-chat-template.jinja`):
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

use common::{V41_SPACED_TEMPLATE, V41_TEMPLATE, post, start, start_templated, start_with};
use serde_json::{Map, Value, json};
use serve::dsml::{ChatParser, Message, Spelling, ToolCall, respell};
use serve::reasoning::ReasoningFormat;
use serve::{ChatTemplate, ScriptedEngine};

const ON: &str = "<｜User｜>hi<｜Assistant｜><think>";
const OFF: &str = "<｜User｜>hi<｜Assistant｜></think>";

const ANSWER: &str = "The user greets me.</think>Hello!";
const PLAIN: &str = "Hello! a</think>b";
const ONE_CALL: &str = "Need the weather.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Seoul</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
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

/// `(name, prompt, format, tools, raw output, expected message)`, the raw output in V4's
/// tags.
fn v4_fixtures() -> Vec<(
    &'static str,
    &'static str,
    ReasoningFormat,
    bool,
    &'static str,
    Message,
)> {
    use ReasoningFormat::{Deepseek, None};
    let weather = || call(0, "get_weather", r#"{"location":"Seoul","days":3}"#);
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
                    r#"{"text":"He said \"hi\" — 서울 ☃\n\\n","filter":{"q":"a\"b","n":[1,2]},"loose":"not json"}"#,
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

/// `text`, written in V4's tags, in `spelling`'s.
fn spelled(text: &str, spelling: Spelling) -> String {
    match spelling {
        Spelling::Joined => text.to_owned(),
        Spelling::Spaced => text
            .replace("｜DSML｜tool_calls", "｜DSML｜ calls")
            .replace("｜DSML｜invoke", "｜DSML｜ invoke")
            .replace("｜DSML｜parameter", "｜DSML｜ parameter"),
    }
}

/// A [`v4_fixtures`] row with its tools flag turned into the spelling the markup is written in.
type SpelledFixture = (
    &'static str,
    &'static str,
    ReasoningFormat,
    Option<Spelling>,
    String,
    Message,
);

/// [`v4_fixtures`] in `spelling`: the raw output, and the expected reasoning and content
/// where they hold the markup as text, respelled; `tools` carries the spelling.
fn fixtures(spelling: Spelling) -> Vec<SpelledFixture> {
    v4_fixtures()
        .into_iter()
        .map(|(name, prompt, format, tools, raw, want)| {
            let want = Message {
                reasoning: spelled(&want.reasoning, spelling),
                content: spelled(&want.content, spelling),
                calls: want.calls,
            };
            (
                name,
                prompt,
                format,
                tools.then_some(spelling),
                spelled(raw, spelling),
                want,
            )
        })
        .collect()
}

const SPELLINGS: [Spelling; 2] = [Spelling::Joined, Spelling::Spaced];

fn streamed(
    prompt: &str,
    format: ReasoningFormat,
    tools: Option<Spelling>,
    pieces: &[&str],
) -> Message {
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
    for (name, prompt, format, tools, raw, want) in SPELLINGS.into_iter().flat_map(fixtures) {
        let got = ChatParser::parse(prompt, format, tools, &raw);
        if got != want {
            wrong.push(format!(
                "{name} ({tools:?}):\n  got  {got:?}\n  want {want:?}"
            ));
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
    for (name, prompt, format, tools, raw, want) in SPELLINGS.into_iter().flat_map(fixtures) {
        let chars: Vec<String> = raw.chars().map(String::from).collect();
        let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            streamed(prompt, format, tools, &pieces),
            want,
            "{name} ({tools:?}): char by char"
        );
        for (at, _) in raw.char_indices().skip(1) {
            let (a, b) = raw.split_at(at);
            assert_eq!(
                streamed(prompt, format, tools, &[a, b]),
                want,
                "{name} ({tools:?}): split at {at}"
            );
        }
    }
}

/// A scan reads the one spelling it is given: the other spelling's block is content, as
/// text, in both directions.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_scan_reads_only_its_own_spelling() {
    for (reads, written) in [
        (Spelling::Joined, Spelling::Spaced),
        (Spelling::Spaced, Spelling::Joined),
    ] {
        let raw = spelled(TWO_CALLS, written);
        let got = ChatParser::parse(OFF, ReasoningFormat::Deepseek, Some(reads), &raw);
        assert!(
            got.calls.is_empty(),
            "{reads:?} read {written:?}'s calls: {got:?}"
        );
        assert_eq!(
            got.content, raw,
            "{reads:?} read {written:?}'s block as text"
        );
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_fixtures_are_the_templates_markup() {
    for (spelling, template) in [
        (Spelling::Joined, V41_TEMPLATE),
        (Spelling::Spaced, V41_SPACED_TEMPLATE),
    ] {
        let t = ChatTemplate::parse(template).expect("template");
        let two_calls = spelled(TWO_CALLS, spelling);
        let m = ChatParser::parse(OFF, ReasoningFormat::Deepseek, Some(spelling), &two_calls);
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
        let want = format!("<｜User｜>hi<｜Assistant｜></think>{two_calls}<｜end▁of▁sentence｜>");
        assert_eq!(rendered, want, "{spelling:?}");
    }
    let t = ChatTemplate::parse(V41_TEMPLATE).expect("template");
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
/// concatenated, one entry per tool call ([`common::fold_call_delta`]).
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
            common::fold_call_delta(&mut calls, tc);
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
    for (spelling, template, script, extra, reasoning) in [
        (Spelling::Joined, V41_TEMPLATE, TWO_CALLS, json!({}), ""),
        (
            Spelling::Joined,
            V41_TEMPLATE,
            ONE_CALL,
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            "Need the weather.",
        ),
        (
            Spelling::Spaced,
            V41_SPACED_TEMPLATE,
            TWO_CALLS,
            json!({}),
            "",
        ),
        (
            Spelling::Spaced,
            V41_SPACED_TEMPLATE,
            ONE_CALL,
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            "Need the weather.",
        ),
    ] {
        let script = spelled(script, spelling);
        let addr = start_templated(Box::new(ScriptedEngine::new(4096, &script)), template);
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

/// Recorded from the server before the split existed (base 6853f02), same request;
/// the goldens' members re-serialized in the order the server builds them
/// (serde_json's `preserve_order`), each value unchanged.
const GOLDEN: &str = r#"{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"abc"}}],"created":_,"model":"m","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":15,"total_tokens":19,"prompt_tokens_details":{"cached_tokens":0}},"id":"_","timings":{_}}"#;
const GOLDEN_STREAM: &str = concat!(
    r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"a"}}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"b"}}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"c"}}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"created":_,"id":"_","model":"m","object":"chat.completion.chunk"}"#,
    "\n\n",
    r#"data: {"choices":[],"created":_,"id":"_","model":"m","object":"chat.completion.chunk","usage":{"completion_tokens":4,"prompt_tokens":15,"total_tokens":19,"prompt_tokens_details":{"cached_tokens":0}},"timings":{_}}"#,
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

// ---------------------------------------------------------------- V4.1's spelling

/// What DeepSeek-V4.1-Flash generated, byte for byte, for a request to write `hello.txt`
/// with the `write` tool in the non-thinking mode: the tags carry a space after the DSML
/// token, which the V4 spelling of the GGUF's own template does not.
const V41_CALL: &str = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"file_path\" string=\"true\">hello.txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\">hi\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

const EOS: &str = "<｜end▁of▁sentence｜>";

fn write_tools() -> Value {
    json!([{"type": "function", "function": {"name": "write", "parameters": {"type": "object",
        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}}}])
}

fn write_user() -> Value {
    json!({"role": "user", "content": "write hello.txt"})
}

/// A model-spelled call parses into `tool_calls`, streamed and whole, to the same message.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_v41_spelled_calls_parse_streamed_and_whole() {
    let addr = start_templated(
        Box::new(ScriptedEngine::new(4096, V41_CALL)),
        V41_SPACED_TEMPLATE,
    );
    let body =
        json!({"model": "m", "messages": [write_user()], "temperature": 0, "tools": write_tools()});
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    let mut whole = c["message"].clone();
    let calls = whole["tool_calls"].as_array().expect("tool_calls").clone();
    assert_eq!(calls.len(), 1, "{v}");
    assert_eq!(calls[0]["function"]["name"], "write");
    let args: Value = serde_json::from_str(calls[0]["function"]["arguments"].as_str().expect("s"))
        .expect("arguments parse");
    assert_eq!(args, json!({"file_path": "hello.txt", "content": "hi\n"}));
    assert_eq!(whole["content"], "", "{v}");
    call_ids(&mut whole);

    let mut sb = body;
    sb["stream"] = json!(true);
    let r = post(addr, "/v1/chat/completions", &sb);
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, mut folded) = fold_stream(&r.body);
    assert_eq!(finish, "tool_calls");
    call_ids(&mut folded);
    assert_eq!(folded, whole, "streamed == non-streamed");
}

/// `/apply-template`'s prompt for `messages` with `tools`.
fn applied(addr: std::net::SocketAddr, messages: &Value, tools: &Value) -> String {
    let r = post(
        addr,
        "/apply-template",
        &json!({"messages": messages, "tools": tools}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    r.json()["prompt"].as_str().expect("prompt").to_owned()
}

/// The server's ids of `text`.
fn ids_of(addr: std::net::SocketAddr, text: &str) -> Vec<u64> {
    let r = post(addr, "/tokenize", &json!({"content": text}));
    assert_eq!(r.status, 200, "{}", r.body);
    r.json()["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|t| t.as_u64().expect("an id"))
        .collect()
}

/// Asserts the prompt of the history `[user, message, tool result]`, the turn `message`
/// answered `write_user()` with, begins with what the engine was fed and generated for
/// it (the first prompt, the model's text `generated`, end of sentence), as text and as
/// the server's ids: the slot's prefix cache keeps the whole turn.
fn assert_history_reuses(
    what: &str,
    addr: std::net::SocketAddr,
    tools: &Value,
    message: &Value,
    generated: &str,
) {
    let first = json!([write_user()]);
    let history = json!([
        write_user(),
        message,
        {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "done"},
    ]);
    let p1 = applied(addr, &first, tools);
    let p2 = applied(addr, &history, tools);
    let fed = format!("{p1}{generated}{EOS}");
    assert!(
        p2.starts_with(&fed),
        "{what}: the history re-renders the turn as the model wrote it:\n  fed  {fed:?}\n  next {p2:?}"
    );
    let want = ids_of(addr, &fed);
    assert_eq!(
        ids_of(addr, &p2)[..want.len()],
        want[..],
        "{what}: the ids of the turn are the next prompt's prefix"
    );
}

/// The message a tool turn answered `write_user()` with, as the server returned it.
fn answered(addr: std::net::SocketAddr, tools: &Value) -> Value {
    let r = post(
        addr,
        "/v1/chat/completions",
        &json!({"model": "m", "messages": [write_user()], "temperature": 0, "tools": tools}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let message = v["choices"][0]["message"].clone();
    assert!(
        message["tool_calls"][0].is_object(),
        "the turn made a call: {v}"
    );
    message
}

/// The V4 template the V4.1 GGUF carries, served spelled as its model writes
/// ([`respell`], what the V4.1 seat does at bind).
fn respelled_template() -> String {
    respell(V41_TEMPLATE, Spelling::Spaced)
        .expect("the V4 template respells")
        .expect("it spells V4's tags")
}

/// A V4.1 GGUF ships V4's template (its tags have no space) but the model writes the
/// spaced tags: served respelled, or under the V4.1 template, a call parses and the
/// history re-renders the turn as the model wrote it, so it keeps the turn's ids.
// PIN(2026-10-19): one gate for the spaced-history prefix; the history-only gate pinned the
// same contract with a hand-made turn and is folded into this one.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_respelled_template_serves_the_models_spelling() {
    for (what, template) in [
        ("the V4 template respelled", respelled_template()),
        ("the V4.1 template", V41_SPACED_TEMPLATE.to_owned()),
    ] {
        let addr = start_templated(Box::new(ScriptedEngine::new(4096, V41_CALL)), &template);
        let tools = write_tools();
        let message = answered(addr, &tools);
        assert_history_reuses(what, addr, &tools, &message, V41_CALL);
    }
}

/// A V4 file's template is served as it is: its calls parse and its history keeps the
/// turn's ids in the joined spelling.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_joined_template_serves_joined_calls() {
    let addr = start_templated(Box::new(ScriptedEngine::new(4096, TWO_CALLS)), V41_TEMPLATE);
    let tools = tools();
    assert_history_reuses("served", addr, &tools, &answered(addr, &tools), TWO_CALLS);
}

/// The spaced template is the joined one with the tags respelled and nothing else:
/// `respell` writes, from the template the V4.1 GGUF carries, the one llama.cpp ships
/// for V4.1, and back; a source that spells no tag, or the spelling asked for, is left
/// alone; a form it does not read is refused by name.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_respell_is_the_released_template_and_back() {
    assert_eq!(
        respell(V41_TEMPLATE, Spelling::Spaced)
            .expect("respells")
            .as_deref(),
        Some(V41_SPACED_TEMPLATE)
    );
    assert_eq!(
        respell(V41_SPACED_TEMPLATE, Spelling::Joined)
            .expect("respells")
            .as_deref(),
        Some(V41_TEMPLATE)
    );
    assert_eq!(Spelling::of_template(V41_TEMPLATE), Spelling::Joined);
    assert_eq!(Spelling::of_template(V41_SPACED_TEMPLATE), Spelling::Spaced);
    let untouched = |source: &str, to| respell(source, to).expect("reads").is_none();
    assert!(untouched(V41_SPACED_TEMPLATE, Spelling::Spaced));
    assert!(untouched(V41_TEMPLATE, Spelling::Joined));
    assert!(untouched(
        include_str!("fixtures/qwen3-chat-template.jinja"),
        Spelling::Spaced
    ));
    assert!(untouched(common::NO_MARKUP_TEMPLATE, Spelling::Spaced));
    // Joined tags built in a form the respelling does not read: a named refusal, never a
    // template whose tags the model does not write served as if they were.
    let odd = "{{- '<' ~ '｜DSML｜' ~ 'tool_calls>' }}";
    let e = respell(odd, Spelling::Spaced).expect_err("refused");
    assert!(e.0.contains("cannot respell"), "{}", e.0);
    // One tag built in another form: the rest respells, and that tag would be left joined.
    // Refused by name, not served half respelled.
    let partial = V41_TEMPLATE.replace("dsml_token + 'invoke", "dsml_token ~ 'invoke");
    assert_ne!(partial, V41_TEMPLATE, "the fixture builds invoke tags");
    let e = respell(&partial, Spelling::Spaced).expect_err("a partial respelling is refused");
    assert!(e.0.contains("cannot respell"), "{}", e.0);
}

/// A request's `reasoning_effort` (what DeepSeek Harness sends) turns V4.1's thinking on, so
/// the reply's reasoning is split out; without it, and with `"none"`, the template's
/// default holds and the whole output is content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_reasoning_effort_turns_thinking_on_for_the_split() {
    let addr = start_with(Box::new(ScriptedEngine::new(4096, ANSWER)));
    let reply = |extra: Value| {
        let mut body = json!({"model": "m", "messages": [write_user()], "temperature": 0});
        body.as_object_mut()
            .expect("an object")
            .extend(extra.as_object().expect("an object").clone());
        let r = post(addr, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["choices"][0]["message"].clone()
    };
    for effort in ["low", "medium", "high", "max"] {
        let m = reply(json!({"reasoning_effort": effort}));
        assert_eq!(
            m["reasoning_content"], "The user greets me.",
            "{effort}: {m}"
        );
        assert_eq!(m["content"], "Hello!", "{effort}: {m}");
    }
    for extra in [json!({}), json!({"reasoning_effort": "none"})] {
        let m = reply(extra.clone());
        assert_eq!(m["content"], ANSWER, "{extra}: {m}");
        assert!(m.get("reasoning_content").is_none(), "{extra}: {m}");
    }
    let m = reply(json!({
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": false},
    }));
    assert_eq!(m["content"], ANSWER, "the explicit switch wins: {m}");
}
