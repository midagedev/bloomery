//! The listening socket of a start, bound before the model is fetched or loaded. Until a server
//! takes it over ([`crate::Server::bind_listener`], [`crate::decide::DecideServer::bind_listener`])
//! it answers every request with a 503 `loading` that says which [`Phase`] the start is in and how
//! far its download is ([`Progress`]), in the error shape llama-server answers while it loads, so a
//! client that reads only the status keeps working. A port that is taken is found here, before the
//! download, and the start is visible from its first second.
//!
//! One accept loop serves the socket throughout: the server's handler is installed into it, so no
//! connection is refused or dropped at the cut-over and the port never unbinds.

use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use serde_json::{Value, json};

use super::{End, accept, error_body, refuse_with, relock};

/// What the 503 says the start is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The model's files are being downloaded.
    Fetch,
    /// The placement is being planned.
    Plan,
    /// The weights are being loaded.
    Load,
}

impl Phase {
    /// The word the 503's `phase` carries.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Phase::Fetch => "fetch",
            Phase::Plan => "plan",
            Phase::Load => "load",
        }
    }
}

/// The message every early answer carries, llama-server's own wording.
const LOADING: &str = "Loading model";

/// What the early answers say.
#[derive(Clone, Copy)]
struct Loading {
    phase: Phase,
    bytes_done: u64,
    bytes_total: u64,
}

/// A connection handler a server installs once.
type Handler = Box<dyn Fn(TcpStream) + Send + Sync>;

/// What the accept loop consults for each connection: the server's handler when one was installed,
/// else the `loading` answer.
struct Gate {
    loading: Mutex<Loading>,
    serve: OnceLock<Handler>,
}

impl Gate {
    fn new() -> Gate {
        Gate {
            loading: Mutex::new(Loading {
                phase: Phase::Fetch,
                bytes_done: 0,
                bytes_total: 0,
            }),
            serve: OnceLock::new(),
        }
    }

    /// One accepted connection: the server's, once it took the listener over, else answered
    /// `loading`.
    fn connection(&self, stream: TcpStream) {
        match self.serve.get() {
            Some(serve) => serve(stream),
            None => refuse_with(stream, &self.body()),
        }
    }

    /// The `loading` answer's body: the status, the phase and the download's bytes beside the error
    /// object.
    fn body(&self) -> Value {
        let loading = *relock(&self.loading);
        let refusal = error_body(503, "unavailable_error", LOADING);
        json!({
            "status": "loading",
            "phase": loading.phase.name(),
            "bytes_done": loading.bytes_done,
            "bytes_total": loading.bytes_total,
            "error": refusal["error"],
        })
    }
}

/// A handle on what the early answers say, for the code that fetches and loads: it moves the phase
/// and the download's bytes while the [`Listener`] is held elsewhere.
#[derive(Clone)]
pub struct Progress(Arc<Gate>);

impl Progress {
    /// Says which phase the start is in.
    pub fn phase(&self, phase: Phase) {
        relock(&self.0.loading).phase = phase;
    }

    /// Says how many of `total` bytes of the download are written.
    pub fn bytes(&self, done: u64, total: u64) {
        let mut loading = relock(&self.0.loading);
        loading.bytes_done = done;
        loading.bytes_total = total;
    }
}

/// A bound listening socket whose server is not up yet. A listener dropped without a server keeps
/// answering `loading` until the process ends.
pub struct Listener {
    local: SocketAddr,
    /// The socket, until its accept loop starts.
    socket: Option<TcpListener>,
    gate: Arc<Gate>,
    /// Where the accept loop reports its end, which the server's own failures share.
    end: mpsc::Sender<End>,
    ended: mpsc::Receiver<End>,
}

impl Listener {
    /// Binds `addr` without accepting: a connection waits in the kernel's queue until a server
    /// takes the listener over.
    pub fn bind(addr: impl ToSocketAddrs) -> io::Result<Listener> {
        let socket = TcpListener::bind(addr)?;
        let local = socket.local_addr()?;
        let (end, ended) = mpsc::channel();
        Ok(Listener {
            local,
            socket: Some(socket),
            gate: Arc::new(Gate::new()),
            end,
            ended,
        })
    }

    /// Binds `addr` and answers every request 503 `loading` until a server takes the listener over.
    pub fn early(addr: impl ToSocketAddrs) -> io::Result<Listener> {
        let mut listener = Listener::bind(addr)?;
        listener.accepting();
        Ok(listener)
    }

    /// The bound address (the real port when bound to port 0).
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// The handle that moves the `loading` answer's phase and bytes.
    #[must_use]
    pub fn progress(&self) -> Progress {
        Progress(Arc::clone(&self.gate))
    }

    /// The channel the server's end is reported on: the accept loop's failure and the server's own.
    pub(crate) fn end(&self) -> mpsc::Sender<End> {
        self.end.clone()
    }

    /// Starts the accept loop, if it is not running: each connection goes to the gate.
    fn accepting(&mut self) {
        if let Some(socket) = self.socket.take() {
            let gate = Arc::clone(&self.gate);
            accept(socket, self.end.clone(), move |stream| {
                gate.connection(stream);
            });
        }
    }

    /// Hands the listener to a server: `serve` answers every connection from here on, through the
    /// loop that already answers `loading` or one started now. Returns the channel the server's end
    /// is read on.
    pub(crate) fn serve(
        mut self,
        serve: impl Fn(TcpStream) + Send + Sync + 'static,
    ) -> mpsc::Receiver<End> {
        let installed = self.gate.serve.set(Box::new(serve));
        assert!(
            installed.is_ok(),
            "a listener is handed over once: `serve` takes it by value"
        );
        self.accepting();
        self.ended
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::time::Duration;

    use super::*;
    use crate::api::testserve::mock_config;
    use crate::{MockEngine, Server, SlotConfig};

    /// One HTTP/1.0 request that waits for its answer at most a few seconds: the status, the head
    /// and the JSON body. A listener that never answers fails the gate by name instead of hanging
    /// it.
    fn ask(addr: SocketAddr, path: &str) -> (u16, String, Value) {
        let mut s = TcpStream::connect(addr).expect("connect");
        s.set_read_timeout(Some(Duration::from_secs(5)))
            .expect("a read timeout");
        write!(s, "GET {path} HTTP/1.0\r\nHost: {addr}\r\n\r\n").expect("write");
        let mut raw = String::new();
        s.read_to_string(&mut raw)
            .unwrap_or_else(|e| panic!("{path}: no answer ({e})"));
        let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .expect("a status line");
        let body = serde_json::from_str(body).unwrap_or_else(|e| panic!("{body:?}: {e}"));
        (status, head.to_owned(), body)
    }

    /// Until a server takes it over, an early listener answers every route 503 with `Retry-After`,
    /// the `loading` status and llama-server's error object, and the phase and bytes follow the
    /// handle.
    #[test]
    fn an_early_listener_answers_every_route_loading() {
        let listener = Listener::early("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr();
        for path in ["/health", "/v1/health", "/v1/models", "/props", "/slots"] {
            let (status, head, v) = ask(addr, path);
            assert_eq!(status, 503, "{path}: {v}");
            assert!(
                head.to_ascii_lowercase().contains("retry-after: 1"),
                "{path}: {head}"
            );
            assert_eq!(v["status"], "loading", "{path}: {v}");
            assert_eq!(v["phase"], "fetch", "{path}: {v}");
            assert_eq!(v["error"]["code"], 503, "{path}: {v}");
            assert_eq!(v["error"]["message"], "Loading model", "{path}: {v}");
            assert_eq!(v["error"]["type"], "unavailable_error", "{path}: {v}");
        }
        let progress = listener.progress();
        progress.phase(Phase::Plan);
        progress.bytes(5, 10);
        let (_, _, v) = ask(addr, "/health");
        assert_eq!(
            (&v["phase"], &v["bytes_done"], &v["bytes_total"]),
            (&json!("plan"), &json!(5), &json!(10)),
            "{v}"
        );
        progress.phase(Phase::Load);
        let (_, _, v) = ask(addr, "/v1/health");
        assert_eq!(v["phase"], "load", "{v}");
    }

    /// A server that takes the listener over answers `/health` 200 from its start on, and not
    /// before: its engine is taken at the bind, yet the listener says `loading` until the server
    /// runs, on the port it already answered at.
    #[test]
    fn a_server_taking_the_listener_over_ends_loading() {
        let listener = Listener::early("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr();
        let server = Server::bind_listener(
            listener,
            Box::new(MockEngine::new(64)),
            mock_config(),
            SlotConfig::default(),
        )
        .unwrap_or_else(|e| panic!("take over: {e}"));
        assert_eq!(server.local_addr().expect("addr"), addr);
        let (status, _, v) = ask(addr, "/health");
        assert_eq!((status, &v["status"]), (503, &json!("loading")), "{v}");
        let (_state, _ended) = server.start().expect("start");
        let (status, _, v) = ask(addr, "/health");
        assert_eq!((status, &v["status"]), (200, &json!("ok")), "{v}");
        let (status, _, v) = ask(addr, "/v1/models");
        assert_eq!(status, 200, "{v}");
    }

    /// A listener bound without accepting serves a server that takes it over, as
    /// `Server::bind_with` binds its own.
    #[test]
    fn a_quiet_listener_serves_the_server_that_takes_it() {
        let listener = Listener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr();
        let server = Server::bind_listener(
            listener,
            Box::new(MockEngine::new(64)),
            mock_config(),
            SlotConfig::default(),
        )
        .unwrap_or_else(|e| panic!("take over: {e}"));
        let (_state, _ended) = server.start().expect("start");
        let (status, _, v) = ask(addr, "/health");
        assert_eq!((status, &v["status"]), (200, &json!("ok")), "{v}");
    }

    /// A port that is taken is refused at the bind, as a taken port: the code an agent reads.
    #[test]
    fn a_taken_port_is_refused_at_the_bind() {
        let held = Listener::early("127.0.0.1:0").expect("bind");
        let Err(e) = Listener::early(held.local_addr()) else {
            panic!("a second listener bound a taken port");
        };
        assert_eq!(e.kind(), io::ErrorKind::AddrInUse, "{e}");
    }
}
