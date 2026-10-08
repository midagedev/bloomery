//! The in-crate tests' server: a mock engine behind a real socket, and one
//! HTTP round trip to it. Tests of `api` and of the modules converted onto it
//! start their servers here, so a field `ServerConfig` gains is set in one
//! place ([`mock_config`]).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, mpsc};

use super::{End, Engine, FATAL_LINGER, Server, ServerConfig, State};
use crate::flag::ApiKeys;

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
        api_keys: ApiKeys::default(),
    }
}

/// A mock server whose key check is on, `keys` set: the server every key
/// test asks a key of.
pub(super) fn spawn_keyed(keys: ApiKeys) -> (SocketAddr, Arc<State>, mpsc::Receiver<End>) {
    let mut config = mock_config();
    config.api_keys = keys;
    spawn(Box::new(crate::MockEngine::new(64)), config)
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
    let (status, _head, body) = roundtrip_head(addr, method, path, headers, body);
    (status, body)
}

/// [`roundtrip`] with the whole reply head beside the body: the status,
/// every header line as sent, and the body — the form a test that pins a
/// response's headers needs.
pub(super) fn roundtrip_head(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (u16, String, String) {
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
    (status, head.to_owned(), body.to_owned())
}

/// The running mock binary, killed when it drops, its bound address read
/// from the startup line and its stderr lines gathered on a channel. The
/// binary as `cargo test -p` builds it (the integration tests' spawns need
/// it too); a run that has not built it (a bare `--lib` after `cargo
/// clean`) fails by name.
pub(super) struct BinServe {
    child: std::process::Child,
    /// The startup line the address was read from.
    pub(super) startup: String,
    pub(super) addr: SocketAddr,
    pub(super) lines: mpsc::Receiver<String>,
}

impl Drop for BinServe {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `bloomery-serve <args>` (which must bind port 0): the one server whose
/// startup line, stderr and `/props` argv are the process's own, which a
/// key must never reach.
pub(super) fn spawn_bin(args: &[&str]) -> BinServe {
    use std::io::BufRead;
    let bin =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/bloomery-serve");
    let mut child = std::process::Command::new(&bin)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
    let err = child.stderr.take().expect("stderr");
    let (tx, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                return;
            }
        }
    });
    let startup = lines
        .recv_timeout(std::time::Duration::from_secs(20))
        .expect("bloomery-serve printed no address");
    let addr = startup
        .rsplit_once("http://")
        .and_then(|(_, a)| a.parse().ok())
        .unwrap_or_else(|| panic!("no address in {startup:?}"));
    BinServe {
        child,
        startup,
        addr,
        lines,
    }
}
