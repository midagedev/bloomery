//! The in-crate tests' server: a mock engine behind a real socket, and one
//! HTTP round trip to it. Tests of `api` and of the modules converted onto it
//! start their servers here, so a field `ServerConfig` gains is set in one
//! place ([`mock_config`]).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, mpsc};

use super::{End, Engine, FATAL_LINGER, Server, ServerConfig, State};

/// The mock's chat template: each message as `<role>\n` and its content,
/// then `<assistant>\n` when a generation prompt is asked for.
pub(super) const TEMPLATE: &str = concat!(
    "{%- for message in messages %}",
    "{{- '<' + message.role + '>\\n' + message.content }}",
    "{%- endfor %}",
    "{%- if add_generation_prompt %}{{- '<assistant>\\n' }}{%- endif %}",
);

/// The configuration every in-crate test server starts from.
pub(super) fn mock_config() -> ServerConfig {
    ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: TEMPLATE.to_owned(),
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
    }
}

/// `engine` served on a free loopback port under `config`: the address, the
/// server's state and its end channel.
pub(super) fn spawn(
    engine: Box<dyn Engine>,
    config: ServerConfig,
) -> (SocketAddr, Arc<State>, mpsc::Receiver<End>) {
    let server = Server::bind("127.0.0.1:0", engine, config).expect("bind");
    let addr = server.local_addr().expect("addr");
    let (state, ended) = server.start().expect("start");
    (addr, state, ended)
}

/// One HTTP/1.0 request with `headers` past the JSON content type, read to
/// the connection's end: the status and the body (a stream's whole body).
pub(super) fn roundtrip(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (u16, String) {
    let mut s = TcpStream::connect(addr).expect("connect");
    let extra: String = headers
        .iter()
        .map(|(k, v)| format!("{k}: {v}\r\n"))
        .collect();
    let req = format!(
        "{method} {path} HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         {extra}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("write");
    let mut raw = String::new();
    s.read_to_string(&mut raw).expect("read");
    let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .expect("a status line");
    (status, body.to_owned())
}
