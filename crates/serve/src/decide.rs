//! A decision model's HTTP server over a [`Decide`]: one prompt pass per request, an answer body
//! out, no generation and no chat.
//!
//! Routes: `POST /v1/systemone` (the request body to [`Decide::decide`]; the answer with a
//! `timings` object appended as its last key), `GET /props` ([`Decide::props`], read once at
//! bind), `GET /health`, `OPTIONS` on any path (the CORS answer [`crate::Server`] gives), and a 404
//! error object for anything else. A refused request is a 400 carrying the decider's message, an
//! engine fault a 500. The decider runs one request at a time behind a mutex; connections are
//! served one thread each, at most [`MAX_CONNECTIONS`] at once, with the same keep-alive and read
//! bounds as [`crate::Server`].

use std::io::{self, BufReader};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};

use crate::MAX_CONNECTIONS;
use crate::api::{JSON, Permit, closing, cors_preflight, error_body, next_request_arrives, refuse};
use crate::http::{self, Request};

/// The commit the build script found (`unknown` for a tree without git), for a decider's
/// `/props`.
pub const COMMIT: &str = env!("BLOOMERY_SERVE_COMMIT");

/// One answered request.
#[derive(Clone, Debug)]
pub struct Decided {
    /// The response JSON object, as the model's reference writes it.
    pub body: String,
    /// The prompt's token count.
    pub prompt_n: usize,
    /// The backbone's prompt pass.
    pub prompt_ms: f64,
    /// Everything after the prompt pass: the head, its rows and the answer body.
    pub head_ms: f64,
}

/// Why a request got no answer.
#[derive(Debug, thiserror::Error)]
pub enum DecideError {
    /// The request itself was refused (a 400): malformed, invalid, or too long.
    #[error("{0}")]
    Refused(String),
    /// The engine failed on a valid request (a 500).
    #[error("{0}")]
    Engine(String),
}

/// A decision model: a request body in, an answer out. The one seam between the server and a
/// model; the model's prompt, head and answer shape stay on its side.
pub trait Decide: Send {
    /// The answer to one request body.
    fn decide(&mut self, body: &str) -> Result<Decided, DecideError>;
    /// What `/props` says about the engine and the model.
    fn props(&self) -> Value;
}

/// One reply, before it is written.
#[derive(Debug, PartialEq)]
struct Reply {
    status: u16,
    ctype: &'static str,
    extra: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn json(status: u16, v: &Value) -> Reply {
        Reply {
            status,
            ctype: JSON,
            extra: Vec::new(),
            body: v.to_string().into_bytes(),
        }
    }

    fn error(status: u16, kind: &str, message: &str) -> Reply {
        Reply::json(status, &error_body(status, kind, message))
    }
}

/// `d`'s body with `timings` appended as its last key; a body that is not a JSON object is an
/// engine fault. The body's own text is kept byte for byte.
fn with_timings(d: &Decided) -> Result<String, DecideError> {
    let not_object = || DecideError::Engine("the decider's answer is not a JSON object".to_owned());
    let parsed: Value = serde_json::from_str(&d.body)
        .map_err(|e| DecideError::Engine(format!("the decider's answer is not JSON: {e}")))?;
    let Value::Object(map) = parsed else {
        return Err(not_object());
    };
    let open = d.body.trim_end().strip_suffix('}').ok_or_else(not_object)?;
    let timings = json!({
        "prompt_n": d.prompt_n,
        "prompt_ms": d.prompt_ms,
        "head_ms": d.head_ms,
        "cache_n": 0,
    });
    let comma = if map.is_empty() { "" } else { "," };
    Ok(format!("{open}{comma}\"timings\":{timings}}}"))
}

/// The reply to one request.
fn reply(decider: &Mutex<Box<dyn Decide>>, props: &Value, req: &Request) -> Reply {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => Reply::json(200, &json!({ "status": "ok" })),
        ("GET", "/props") => Reply::json(200, props),
        ("OPTIONS", _) => Reply {
            status: 204,
            ctype: "text/plain",
            extra: cors_preflight().to_vec(),
            body: Vec::new(),
        },
        ("POST", "/v1/systemone") => {
            let Ok(body) = std::str::from_utf8(&req.body) else {
                return Reply::error(
                    400,
                    "invalid_request_error",
                    "the request body is not UTF-8",
                );
            };
            let Ok(mut d) = decider.lock() else {
                return Reply::error(
                    500,
                    "server_error",
                    "the decider panicked in an earlier request",
                );
            };
            match d.decide(body).and_then(|a| with_timings(&a)) {
                Ok(text) => Reply {
                    status: 200,
                    ctype: JSON,
                    extra: Vec::new(),
                    body: text.into_bytes(),
                },
                Err(DecideError::Refused(m)) => Reply::error(400, "invalid_request_error", &m),
                Err(DecideError::Engine(m)) => Reply::error(500, "server_error", &m),
            }
        }
        _ => Reply::error(404, "not_found_error", "File Not Found"),
    }
}

/// A bound, not yet running decision server.
pub struct DecideServer {
    listener: TcpListener,
    decider: Arc<Mutex<Box<dyn Decide>>>,
    props: Arc<Value>,
}

impl DecideServer {
    /// Binds `addr` and takes ownership of the decider; `/props` is read from it here.
    pub fn bind(addr: impl ToSocketAddrs, decider: Box<dyn Decide>) -> io::Result<DecideServer> {
        let listener = TcpListener::bind(addr)?;
        let props = Arc::new(decider.props());
        Ok(DecideServer {
            listener,
            decider: Arc::new(Mutex::new(decider)),
            props,
        })
    }

    /// The bound address (the real port when bound to port 0).
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accepts connections until `accept` fails, and returns that error.
    pub fn run(self) -> io::Error {
        let live = Arc::new(AtomicUsize::new(0));
        for conn in self.listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => return e,
            };
            let Some(permit) = Permit::take(&live) else {
                refuse(
                    stream,
                    &format!("the server is serving {MAX_CONNECTIONS} connections, its limit"),
                );
                continue;
            };
            let (decider, props) = (Arc::clone(&self.decider), Arc::clone(&self.props));
            let spawned = thread::Builder::new()
                .name("decide".to_owned())
                .spawn(move || {
                    let _permit = permit;
                    serve_conn(&decider, &props, stream);
                });
            if let Err(e) = spawned {
                eprintln!("bloomery-serve: cannot start a connection thread: {e}");
            }
        }
        io::Error::other("the listener's accept loop ended")
    }
}

fn serve_conn(decider: &Mutex<Box<dyn Decide>>, props: &Value, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let Ok(mut w) = stream.try_clone() else {
        return;
    };
    let mut r = BufReader::new(stream);
    loop {
        if !next_request_arrives(&mut r) {
            return;
        }
        let req = match http::read_request(&mut r, &mut w) {
            Ok(Some(req)) => req,
            Ok(None) => return,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                let a = Reply::error(400, "invalid_request_error", &e.to_string());
                let _ = http::respond(&mut w, &closing(), a.status, a.ctype, &a.extra, &a.body);
                return;
            }
            Err(_) => return,
        };
        let a = reply(decider, props, &req);
        if http::respond(&mut w, &req, a.status, a.ctype, &a.extra, &a.body).is_err()
            || !req.keep_alive
        {
            return;
        }
    }
}
