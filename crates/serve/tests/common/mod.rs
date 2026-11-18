//! Shared harness for the server gates: a mock-engine server on a free port and
//! an HTTP/1.0 client (the server closes after each response, so the body is
//! whatever arrives before EOF and no de-chunking is needed).

#![allow(dead_code, reason = "each gate uses a different subset")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};
use serve::dsml::{ChatParser, MarkupError, Message, ToolCall, Tools};
use serve::glmxml::{ArgTypes, KvSpelling};
use serve::reasoning::ReasoningFormat;
use serve::{Engine, FATAL_LINGER, MockEngine, Server, ServerConfig};

pub const V41_TEMPLATE: &str = include_str!("../fixtures/v41-chat-template.jinja");
/// The same template with the V4.1 release's DSML spelling (a space after the token).
pub const V41_SPACED_TEMPLATE: &str =
    include_str!("../fixtures/v41-dsml-spaced-chat-template.jinja");
pub const FIELDS: &str = include_str!("../fixtures/llama-server-fields.json");

/// A synthetic chat template that teaches no tool-call markup the server
/// parses: its source spells none of the families' tags and a rendered
/// assistant call carries no `<tool_call>` the probe could read, so it types
/// [`serve::dsml::ToolFormat::Unparsed`]. No real model's template is left in
/// that state — the refusal's own contract needs one to refuse.
pub const NO_MARKUP_TEMPLATE: &str = concat!(
    "{%- for message in messages %}",
    "{{- '<' + message.role + '>\\n' + message.content }}",
    "{%- endfor %}",
    "{%- if add_generation_prompt %}{{- '<assistant>\\n' }}{%- endif %}",
);

/// Starts a server on the mock engine (context `ctx`) and the V4.1 template.
pub fn start(ctx: usize) -> SocketAddr {
    start_with(Box::new(MockEngine::new(ctx)))
}

/// Starts a server on `engine` and the V4.1 template.
pub fn start_with(engine: Box<dyn Engine>) -> SocketAddr {
    start_templated(engine, V41_TEMPLATE)
}

/// Starts a server on `engine` and the chat template `template`; a template
/// the parser refuses panics with the parser's error.
pub fn start_templated(engine: Box<dyn Engine>, template: &str) -> SocketAddr {
    start_configured(engine, template, None)
}

/// Starts a server on `engine` and the V4.1 template that saves and restores
/// slots in `dir`.
pub fn start_slots(engine: Box<dyn Engine>, dir: &Path) -> SocketAddr {
    start_configured(engine, V41_TEMPLATE, Some(dir.to_path_buf()))
}

/// Starts a server on the mock engine (context `ctx`) and the V4.1 template whose 501 answers name
/// the flags `starts` holds.
pub fn start_starts(ctx: usize, starts: serve::flag::Starts) -> SocketAddr {
    let config = ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: V41_TEMPLATE.to_owned(),
        chat_preparation: None,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    Server::bind("127.0.0.1:0", Box::new(MockEngine::new(ctx)), config)
        .unwrap_or_else(|e| panic!("bind: {e}"))
        .with_starts(starts)
        .spawn()
        .expect("spawn")
}

fn start_configured(
    engine: Box<dyn Engine>,
    template: &str,
    slot_save_path: Option<PathBuf>,
) -> SocketAddr {
    let config = ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: template.to_owned(),
        chat_preparation: None,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let server =
        Server::bind("127.0.0.1:0", engine, config).unwrap_or_else(|e| panic!("bind: {e}"));
    server.spawn().expect("spawn")
}

/// A fresh, empty directory under the system temp dir, named for `test` and
/// this process, so gates running in parallel never share one.
pub fn fresh_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bloomery-serve-{}-{test}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

/// Removes a [`fresh_dir`] once its gate passed; a failed gate leaves it for
/// reading.
pub fn drop_dir(dir: &Path) {
    std::fs::remove_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

pub struct Reply {
    pub status: u16,
    pub head: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The `data:` payloads of an SSE body, in order.
    pub fn events(&self) -> Vec<String> {
        self.body
            .split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .map(str::to_owned)
            .collect()
    }
}

/// One request over a fresh HTTP/1.0 connection.
pub fn call(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Reply {
    call_headers(addr, method, path, &[], body)
}

/// One HTTP/1.0 request with additional headers over a fresh connection.
pub fn call_headers(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> Reply {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(60)))
        .expect("timeout");
    let body = body.unwrap_or("");
    let extra: String = headers
        .iter()
        .map(|(k, v)| format!("{k}: {v}\r\n"))
        .collect();
    let req = format!(
        "{method} {path} HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("write");
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read");
    let raw = String::from_utf8(raw).expect("UTF-8 response");
    let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .expect("status line");
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    Reply {
        status,
        head: head.to_owned(),
        headers,
        body: body.to_owned(),
    }
}

pub fn post(addr: SocketAddr, path: &str, body: &Value) -> Reply {
    call(addr, "POST", path, Some(&body.to_string()))
}

pub fn get(addr: SocketAddr, path: &str) -> Reply {
    call(addr, "GET", path, None)
}

/// The fixture's key list under `section`.
pub fn fixture_keys(section: &str) -> Vec<String> {
    let v: Value = serde_json::from_str(FIELDS).expect("fixture JSON");
    v[section]
        .as_array()
        .unwrap_or_else(|| panic!("fixture section {section}"))
        .iter()
        .map(|k| k.as_str().expect("key").to_owned())
        .collect()
}

/// Asserts every fixture key of `section` is present in `obj`.
pub fn assert_keys(obj: &Value, section: &str) {
    for k in fixture_keys(section) {
        assert!(obj.get(&k).is_some(), "{section}: missing `{k}` in {obj}");
    }
}

/// Starts a server on `engine` and the chat template `template`, its reasoning split on the
/// model's own `tags` ([`Server::with_think_tags`]).
pub fn start_tagged(
    engine: Box<dyn Engine>,
    template: &str,
    tags: serve::reasoning::ThinkTags,
) -> SocketAddr {
    let config = ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: template.to_owned(),
        chat_preparation: None,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let server = Server::bind("127.0.0.1:0", engine, config)
        .unwrap_or_else(|e| panic!("bind: {e}"))
        .with_think_tags(tags);
    server.spawn().expect("spawn")
}

/// Starts a server of `parallel` slots on `engine`, the V4.1 template and the
/// sampler factory `sampler`.
pub fn start_sampling(
    engine: Box<dyn Engine>,
    parallel: usize,
    sampler: serve::SamplerFactory,
) -> SocketAddr {
    let config = ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: V41_TEMPLATE.to_owned(),
        chat_preparation: None,
        sampler: Some(sampler),
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let slots = serve::SlotConfig {
        parallel,
        ..serve::SlotConfig::default()
    };
    let server = Server::bind_with("127.0.0.1:0", engine, config, slots)
        .unwrap_or_else(|e| panic!("bind: {e}"));
    server.spawn().expect("spawn")
}

/// Folds one streamed `tool_calls` delta into `calls`, one OpenAI tool call
/// (`id`, `type`, `function`) each. The delta that names a call opens it, at
/// the next index and with no arguments yet; every other delta is an
/// `arguments` fragment of the call it indexes, the last one open.
pub fn fold_call_delta(calls: &mut Vec<Value>, tc: &Value) {
    let index = usize::try_from(tc["index"].as_u64().expect("index")).expect("small");
    let args = tc["function"]["arguments"].as_str().expect("arguments");
    if tc.get("id").is_some() {
        assert_eq!(index, calls.len(), "each call opens once, in order: {tc}");
        assert_eq!(tc["type"], "function", "{tc}");
        assert_eq!(args, "", "the opening delta carries no arguments: {tc}");
        calls.push(json!({
            "id": tc["id"],
            "type": "function",
            "function": { "name": tc["function"]["name"], "arguments": "" },
        }));
    } else {
        assert_eq!(
            index + 1,
            calls.len(),
            "a fragment extends the open call: {tc}"
        );
        assert!(tc["function"].get("name").is_none(), "{tc}");
        let open = &mut calls[index]["function"]["arguments"];
        *open = json!(format!("{}{args}", open.as_str().expect("text")));
    }
}

/// The ways to feed `raw` to a streaming parser: the whole text at once, a
/// character at a time, split in two at every character, and split in three
/// at every pair of tag boundaries (the byte before a `<` and the byte after
/// a `>`). `tag_stride` keeps every `tag_stride`th boundary, so 1 is every
/// pair and a larger stride thins the pairs of a text with many tags.
pub fn every_chunking(raw: &str, tag_stride: usize) -> Vec<Vec<&str>> {
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
        .step_by(tag_stride)
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

/// `body` with one more tool, whose `tags` array schema the structured-output
/// subset cannot compile: `tool_choice` auto then leaves the reply to the
/// model (no watch), so the markup a script writes reaches the scan as it is.
pub fn unwatched(mut body: Value) -> Value {
    body["tools"]
        .as_array_mut()
        .expect("tools")
        .push(json!({"type": "function", "function": {"name": "note",
            "parameters": {"type": "object", "properties": {"tags": {"type": "array"}}}}}));
    body
}

/// A parsed call: the `index`-th of its message, `name` with `arguments` (JSON text).
pub fn tool_call(index: usize, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        index,
        name: name.to_owned(),
        arguments: arguments.to_owned(),
    }
}

/// A parsed message of `reasoning`, `content` and `calls`.
pub fn msg(reasoning: &str, content: &str, calls: Vec<ToolCall>) -> Message {
    Message {
        reasoning: reasoning.to_owned(),
        content: content.to_owned(),
        calls,
    }
}

/// The message `p` reads from `pieces` pushed in order and the finish, summed from the deltas
/// each push and the finish return; the sum is the parser's own message.
pub fn sum_deltas(p: &mut ChatParser, pieces: &[&str]) -> Result<Message, MarkupError> {
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

/// A chat request at temperature 0 that lists `tools`, streamed or not.
pub fn tool_body(tools: Value, stream: bool) -> Value {
    json!({
        "model": "m",
        "messages": [{"role": "user", "content": "weather and time?"}],
        "temperature": 0,
        "tools": tools,
        "stream": stream,
    })
}

/// A streamed chat body folded back into `(finish_reason, message)`, call ids left out; `Err`
/// with its message when the stream ended on an error event. A `delta.content` that holds any of
/// `markup` (the spelling's call tags) is a failure: markup never reaches the content.
pub fn fold_stream(body: &str, markup: &[&str]) -> Result<(Value, Value), String> {
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
                !markup.iter().any(|m| s.contains(m)),
                "markup reached delta.content: {s}"
            );
            content.push_str(s);
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            fold_call_delta(&mut calls, tc);
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

/// A table's wrong cases, listed together: a failure names every one.
fn none_wrong(what: &str, wrong: &[String]) {
    assert!(
        wrong.is_empty(),
        "{} {what} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Every fixture `(name, prompt, format, raw, want)` read whole by `parse` (the prompt, the
/// format and the pieces) is `want`, each of its calls' arguments a JSON object.
pub fn check_fixtures<P: AsRef<str>, R: AsRef<str>>(
    fixtures: Vec<(&str, P, ReasoningFormat, R, Message)>,
    parse: impl Fn(&str, ReasoningFormat, &[&str]) -> Result<Message, MarkupError>,
) {
    let mut wrong = Vec::new();
    for (name, prompt, format, raw, want) in fixtures {
        match parse(prompt.as_ref(), format, &[raw.as_ref()]) {
            Ok(got) if got == want => {
                for c in &got.calls {
                    let v: Value = serde_json::from_str(&c.arguments).expect("arguments are JSON");
                    assert!(v.is_object(), "{name}: arguments are an object");
                }
            }
            got => wrong.push(format!("{name}:\n  got  {got:?}\n  want {want:?}")),
        }
    }
    none_wrong("fixture(s)", &wrong);
}

/// Every malformed case `(name, raw, want)` read whole by `parse` is the named error `want`,
/// never content and never a guessed call.
pub fn check_malformed<R: AsRef<str>, E: Clone + std::fmt::Debug + Into<MarkupError>>(
    cases: Vec<(&str, R, E)>,
    parse: impl Fn(&[&str]) -> Result<Message, MarkupError>,
) {
    let mut wrong = Vec::new();
    for (name, raw, want) in cases {
        match parse(&[raw.as_ref()]) {
            Err(e) if e == want.clone().into() => {}
            got => wrong.push(format!("{name}:\n  got  {got:?}\n  want Err({want:?})")),
        }
    }
    none_wrong("case(s)", &wrong);
}

/// A message, or an error, never depends on how the text was cut ([`every_chunking`] at a tag
/// stride of 1): every fixture, and every malformed case under `on` (the prompt and the format
/// its errors are read at), reads the same through `parse` whole and cut.
pub fn check_chunking<P: AsRef<str>, R: AsRef<str>, M: AsRef<str>, E: Into<MarkupError>>(
    fixtures: Vec<(&str, P, ReasoningFormat, R, Message)>,
    malformed: Vec<(&str, M, E)>,
    on: (&str, ReasoningFormat),
    parse: impl Fn(&str, ReasoningFormat, &[&str]) -> Result<Message, MarkupError>,
) {
    let check = |name: &str, (prompt, format): (&str, ReasoningFormat), raw: &str, want| {
        for pieces in every_chunking(raw, 1) {
            assert_eq!(parse(prompt, format, &pieces), want, "{name}: {pieces:?}");
        }
    };
    for (name, prompt, format, raw, want) in fixtures {
        check(name, (prompt.as_ref(), format), raw.as_ref(), Ok(want));
    }
    for (name, raw, e) in malformed {
        check(name, on, raw.as_ref(), Err(e.into()));
    }
}

/// The scripted server at `addr`, asked the tool request `body(stream)`, answers the markup it
/// cannot parse with the named error `want`: a 500 without a stream, and an error event that ends
/// a stream (`fold`, the spelling's [`fold_stream`]) on a 200 with one.
pub fn check_api_error(
    addr: SocketAddr,
    body: impl Fn(bool) -> Value,
    want: &str,
    fold: impl Fn(&str) -> Result<(Value, Value), String>,
) {
    let r = post(addr, "/v1/chat/completions", &body(false));
    assert_eq!(r.status, 500, "{}", r.body);
    let m = r.json()["error"]["message"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    assert!(m.starts_with(want), "{m}");
    let r = post(addr, "/v1/chat/completions", &body(true));
    assert_eq!(r.status, 200, "{}", r.body);
    let e = fold(&r.body).expect_err("an error event");
    assert!(e.starts_with(want), "{e}");
}

/// The tools the key/value spellings' fixtures call: `get_weather` (a string `location`, an
/// integer `days`), `get_time` (no arguments) and `save_note` (an object `filter`, a string
/// `text`).
pub fn kv_tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "save_note", "parameters": {"type": "object",
            "properties": {"filter": {"type": "object"}, "text": {"type": "string"}}}}},
    ])
}

/// The key/value markup scan of `spelling` over [`kv_tools`]' argument types.
pub fn kv_scan(spelling: KvSpelling) -> Option<Tools> {
    Some(Tools::KvMarkup(
        spelling,
        ArgTypes::of_tools(Some(&kv_tools())),
    ))
}
