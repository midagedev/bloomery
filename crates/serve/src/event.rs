//! The events of a start, one JSON object a line on stdout, for the agent or supervisor that
//! started the server: `bound` (the listener is up and answers `loading`), `ready` (the first
//! moment requests are served, the line to wait on), `error` (before every non-zero exit),
//! `stopped` (the server ended), and `pick` (a start by name's choice, built by the caller). Every
//! line names [`SCHEMA`], so a reader can refuse a version it does not know. The human-readable
//! lines stay on stderr.
//!
//! The library prints `ready` and `stopped` for the server that runs, so no seat adds code for
//! them; a process that wants them calls [`enable`] first, and a process that does not (a test,
//! another server) prints nothing here.

use std::io::{self, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, Value, json};

use crate::exit::Code;

/// The schema every line names.
pub const SCHEMA: &str = "bloomery.event/1";

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Turns the events on for this process: [`emit`] prints from here on. A binary calls it first
/// thing in `main`.
pub fn enable() {
    ENABLED.store(true, Ordering::SeqCst);
}

/// Why a running server stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The first `SIGINT` or `SIGTERM`: the orderly stop.
    Signal,
    /// A `POST /shutdown`: the orderly stop.
    Shutdown,
    /// The engine failed.
    Engine,
    /// The listener failed.
    Listener,
}

impl Cause {
    /// The word the `stopped` event carries.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Cause::Signal => "signal",
            Cause::Shutdown => "shutdown",
            Cause::Engine => "engine",
            Cause::Listener => "listener",
        }
    }
}

/// One event.
pub enum Event<'a> {
    /// A start by name's choice: `fields` are the line's keys after `schema` and `event`.
    Pick(&'a Map<String, Value>),
    /// The listener is up at `url` and answers `loading`.
    Bound {
        /// Where the server listens.
        url: &'a str,
    },
    /// The server serves requests.
    Ready {
        /// Where the server listens.
        url: &'a str,
        /// The model's name, as `/v1/models` lists it.
        model: &'a str,
        /// The context each slot serves.
        ctx: u64,
        /// The slots served at once.
        slots: u64,
        /// The placement's word, when the seat names one.
        place: Option<&'a str>,
    },
    /// The process exits with `code`'s status after this line; `next` are argument lists to run
    /// instead.
    Error {
        /// The refusal's class.
        code: Code,
        /// What was refused and why.
        message: &'a str,
        /// Argument lists that fix the refusal, each one runnable as it stands.
        next: &'a [&'a [&'a str]],
    },
    /// The server ended; `exit` is the status the process leaves with.
    Stopped {
        /// Why it stopped.
        cause: Cause,
        /// The status the process leaves with.
        exit: Code,
    },
}

/// The line of `event`, without its newline.
#[must_use]
pub fn render(event: &Event<'_>) -> String {
    let mut line = Map::new();
    line.insert("schema".to_owned(), json!(SCHEMA));
    let mut put = |key: &str, value: Value| {
        line.insert(key.to_owned(), value);
    };
    match event {
        Event::Pick(fields) => {
            put("event", json!("pick"));
            for (key, value) in fields.iter() {
                put(key, value.clone());
            }
        }
        Event::Bound { url } => {
            put("event", json!("bound"));
            put("url", json!(url));
            put("phase", json!("loading"));
        }
        Event::Ready {
            url,
            model,
            ctx,
            slots,
            place,
        } => {
            put("event", json!("ready"));
            put("url", json!(url));
            put("model", json!(model));
            put("ctx", json!(ctx));
            put("slots", json!(slots));
            put("place", json!(place));
        }
        Event::Error {
            code,
            message,
            next,
        } => {
            put("event", json!("error"));
            put("exit", json!(code.status()));
            put("code", json!(code.name()));
            put("message", json!(message));
            put("hint", json!(code.hint()));
            put("next", json!(next));
        }
        Event::Stopped { cause, exit } => {
            put("event", json!("stopped"));
            put("cause", json!(cause.name()));
            put("exit", json!(exit.status()));
        }
    }
    Value::Object(line).to_string()
}

/// Prints `event` as one line on stdout, if the process [`enable`]d the events. A stdout that
/// cannot be written (a reader that went away) is named on stderr and the server goes on.
pub fn emit(event: &Event<'_>) {
    if !ENABLED.load(Ordering::SeqCst) {
        return;
    }
    let line = render(event);
    let mut out = io::stdout().lock();
    if let Err(e) = writeln!(out, "{line}").and_then(|()| out.flush()) {
        eprintln!("bloomery-serve: stdout: {e}");
    }
}

/// The URL a client reaches a server bound to `addr` at: an unspecified address (`0.0.0.0`, `::`)
/// is reached through its loopback.
#[must_use]
pub fn url(addr: SocketAddr) -> String {
    let ip = match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        IpAddr::V4(ip) => IpAddr::V4(ip),
        IpAddr::V6(ip) => IpAddr::V6(ip),
    };
    format!("http://{}", SocketAddr::new(ip, addr.port()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(event: &Event<'_>) -> Value {
        serde_json::from_str(&render(event)).expect("a line is JSON")
    }

    /// Every line opens with its schema and its event, then its own keys in the order the contract
    /// lists them.
    #[test]
    fn a_line_names_its_schema_and_event_first() {
        let line = render(&Event::Bound {
            url: "http://127.0.0.1:8080",
        });
        assert_eq!(
            line,
            r#"{"schema":"bloomery.event/1","event":"bound","url":"http://127.0.0.1:8080","phase":"loading"}"#
        );
        let line = render(&Event::Ready {
            url: "http://127.0.0.1:8080",
            model: "m",
            ctx: 4096,
            slots: 2,
            place: None,
        });
        assert_eq!(
            line,
            r#"{"schema":"bloomery.event/1","event":"ready","url":"http://127.0.0.1:8080","model":"m","ctx":4096,"slots":2,"place":null}"#
        );
        let line = render(&Event::Stopped {
            cause: Cause::Engine,
            exit: Code::Engine,
        });
        assert_eq!(
            line,
            r#"{"schema":"bloomery.event/1","event":"stopped","cause":"engine","exit":70}"#
        );
    }

    /// An `error` carries the code's status, name and hint and the argument lists as given; a
    /// `pick` carries the caller's keys after the two every line has.
    #[test]
    fn error_and_pick_lines_carry_their_fields() {
        let v = parsed(&Event::Error {
            code: Code::PortInUse,
            message: "port 8080 is taken",
            next: &[&["--port", "0"], &["--port", "8081"]],
        });
        assert_eq!(v["exit"], 83, "{v}");
        assert_eq!(v["code"], "port-in-use", "{v}");
        assert_eq!(v["message"], "port 8080 is taken", "{v}");
        assert_eq!(v["hint"], Code::PortInUse.hint(), "{v}");
        assert_eq!(
            v["next"],
            json!([["--port", "0"], ["--port", "8081"]]),
            "{v}"
        );

        let mut fields = Map::new();
        fields.insert("variant".to_owned(), json!("q4_k_m"));
        let v = parsed(&Event::Pick(&fields));
        assert_eq!(v["event"], "pick", "{v}");
        assert_eq!(v["variant"], "q4_k_m", "{v}");
        assert_eq!(v["schema"], SCHEMA, "{v}");
    }

    /// A server bound to every address is reached through its loopback.
    #[test]
    fn an_unspecified_address_is_reached_through_loopback() {
        let at = |s: &str| url(s.parse().expect("an address"));
        assert_eq!(at("127.0.0.1:8080"), "http://127.0.0.1:8080");
        assert_eq!(at("0.0.0.0:9000"), "http://127.0.0.1:9000");
        assert_eq!(at("[::]:9000"), "http://[::1]:9000");
        assert_eq!(at("192.168.1.5:80"), "http://192.168.1.5:80");
    }
}
