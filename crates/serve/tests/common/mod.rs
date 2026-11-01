//! Shared harness for the server gates: a mock-engine server on a free port and
//! an HTTP/1.0 client (the server closes after each response, so the body is
//! whatever arrives before EOF and no de-chunking is needed).

#![allow(dead_code, reason = "each gate uses a different subset")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};
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
