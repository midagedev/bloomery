//! Shared harness for the server gates: a mock-engine server on a free port and
//! an HTTP/1.0 client (the server closes after each response, so the body is
//! whatever arrives before EOF and no de-chunking is needed).

#![allow(dead_code, reason = "each gate uses a different subset")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::Value;
use serve::{Engine, FATAL_LINGER, MockEngine, Server, ServerConfig};

pub const V41_TEMPLATE: &str = include_str!("../fixtures/v41-chat-template.jinja");
pub const FIELDS: &str = include_str!("../fixtures/llama-server-fields.json");

/// Starts a server on the mock engine (context `ctx`) and the V4.1 template.
pub fn start(ctx: usize) -> SocketAddr {
    start_with(Box::new(MockEngine::new(ctx)))
}

/// Starts a server on `engine` and the V4.1 template.
pub fn start_with(engine: Box<dyn Engine>) -> SocketAddr {
    let config = ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: V41_TEMPLATE.to_owned(),
        sampler: None,
        fatal_linger: FATAL_LINGER,
    };
    let server = Server::bind("127.0.0.1:0", engine, config).expect("bind");
    server.spawn().expect("spawn")
}

pub struct Reply {
    pub status: u16,
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
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(60)))
        .expect("timeout");
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
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
