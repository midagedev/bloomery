//! Gate: the start contract of `bloomery-serve` over the mock engine — what an agent that starts
//! the server reads. The events are JSON lines on stdout (`bound`, `ready`, `error`); before
//! `ready` every route answers 503 `loading` and after it `/health` answers 200; a port that is
//! taken exits 83 before the load; `--help` exits 0 with the usage on stdout; a refusal exits with
//! its code and prints one `error` event. Each clause runs the mock binary as a process, as the
//! agent does.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

#[path = "common/process.rs"]
mod process;

use process::Reaped;

const MOCK: &str = env!("CARGO_BIN_EXE_bloomery-serve");

/// The mock's load, long enough that a client reaches it while it loads and short enough for a
/// gate.
const LOAD_MS: u64 = 4000;

/// Starts the mock with `args`; returns it and a channel of its stdout lines. Its stderr is not
/// read: the events carry what the clauses assert.
fn spawn(args: &[&str]) -> (Reaped, mpsc::Receiver<String>) {
    let mut child = Command::new(MOCK)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bloomery-serve");
    let out = child.stdout.take().expect("a piped stdout");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                return;
            }
        }
    });
    (Reaped(child), rx)
}

/// The next stdout line, which must be an event of kind `kind` and arrive within `within`.
fn next_event(lines: &mpsc::Receiver<String>, kind: &str, within: Duration) -> Value {
    let line = lines
        .recv_timeout(within)
        .unwrap_or_else(|e| panic!("no `{kind}` event within {within:?}: {e}"));
    let v: Value =
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("not JSON ({e}): {line:?}"));
    assert_eq!(v["schema"], "bloomery.event/1", "{line}");
    assert_eq!(v["event"], kind, "expected a `{kind}` event: {line}");
    v
}

/// Where the event's `url` points.
fn addr_of(event: &Value) -> SocketAddr {
    event["url"]
        .as_str()
        .and_then(|u| u.strip_prefix("http://"))
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| panic!("no address in {event}"))
}

/// One HTTP/1.0 GET that waits for its answer at most a few seconds: the status and the JSON body.
/// A server that accepts and never answers is an error here, not a hang.
fn get(addr: SocketAddr, path: &str) -> Result<(u16, Value), String> {
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .map_err(|e| format!("connect: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| format!("timeout: {e}"))?;
    write!(s, "GET {path} HTTP/1.0\r\nHost: {addr}\r\n\r\n").map_err(|e| format!("write: {e}"))?;
    let mut raw = String::new();
    s.read_to_string(&mut raw)
        .map_err(|e| format!("no answer: {e}"))?;
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("no head in {raw:?}"))?;
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| format!("no status in {head:?}"))?;
    let body = serde_json::from_str(body).map_err(|e| format!("body {body:?}: {e}"))?;
    Ok((status, body))
}

/// During the load every route answers 503 `loading`, in llama-server's error shape, naming the
/// phase.
fn assert_loading(addr: SocketAddr) {
    for path in ["/health", "/v1/health", "/v1/models"] {
        let (status, v) = get(addr, path).unwrap_or_else(|e| panic!("{path} during the load: {e}"));
        assert_eq!(status, 503, "{path}: {v}");
        assert_eq!(v["status"], "loading", "{path}: {v}");
        assert_eq!(v["phase"], "load", "{path}: {v}");
        assert_eq!(v["error"]["code"], 503, "{path}: {v}");
        assert_eq!(v["error"]["message"], "Loading model", "{path}: {v}");
        assert_eq!(v["error"]["type"], "unavailable_error", "{path}: {v}");
    }
}

/// `bound` opens the start and the listener answers `loading` from it on; `ready` comes once, after
/// the load, and from it on `/health` answers 200.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ready_comes_once_after_the_load_and_health_follows_it() {
    let (mut mock, lines) = spawn(&["--port", "0", "--mock-load-ms", &LOAD_MS.to_string()]);
    let bound = next_event(&lines, "bound", Duration::from_secs(20));
    let bound_at = Instant::now();
    assert_eq!(bound["phase"], "loading", "{bound}");
    let addr = addr_of(&bound);
    assert_loading(addr);

    let ready = next_event(&lines, "ready", Duration::from_secs(30));
    assert!(
        bound_at.elapsed() >= Duration::from_millis(LOAD_MS / 2),
        "`ready` came {:?} after `bound`, before the {LOAD_MS} ms load ended: {ready}",
        bound_at.elapsed()
    );
    assert_eq!(ready["url"], bound["url"], "{ready}");
    assert_eq!(
        (
            &ready["model"],
            &ready["ctx"],
            &ready["slots"],
            &ready["place"]
        ),
        (&json!("mock"), &json!(4096), &json!(1), &Value::Null),
        "{ready}"
    );
    for _ in 0..20 {
        let (status, v) =
            get(addr, "/health").unwrap_or_else(|e| panic!("/health after ready: {e}"));
        assert_eq!((status, &v["status"]), (200, &json!("ok")), "{v}");
    }

    mock.0.kill().expect("kill the mock");
    mock.0.wait().expect("reap the mock");
    let rest: Vec<String> = lines.iter().collect();
    assert!(
        rest.iter().all(|l| !l.contains(r#""event":"ready""#)),
        "`ready` was printed again: {rest:?}"
    );
}

/// A port that is taken is refused at the listener's bind, before the load: the process exits 83
/// with an `error` event while the load it would have run is still far off.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_port_in_use_exits_83_before_the_load_ends() {
    let held = TcpListener::bind("127.0.0.1:0").expect("hold a port");
    let port = held.local_addr().expect("the held address").port();
    let (mut mock, lines) = spawn(&["--port", &port.to_string(), "--mock-load-ms", "20000"]);
    let until = Instant::now() + Duration::from_secs(8);
    let status = loop {
        if let Some(status) = mock.0.try_wait().expect("wait on the mock") {
            break status;
        }
        assert!(
            Instant::now() < until,
            "the mock still ran 8 s after its port was taken: it binds after the load"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(83), "{status:?}");
    let error = next_event(&lines, "error", Duration::from_secs(5));
    assert_eq!(
        (&error["exit"], &error["code"], &error["next"]),
        (&json!(83), &json!("port-in-use"), &json!([["--port", "0"]])),
        "{error}"
    );
    assert!(
        error["hint"].as_str().is_some_and(|h| !h.is_empty()),
        "{error}"
    );
}

/// `--help` is a success: status 0, the usage on stdout, nothing on stderr.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_help_exits_zero_with_the_usage_on_stdout() {
    for flag in ["--help", "-h"] {
        let out = Command::new(MOCK)
            .arg(flag)
            .output()
            .expect("run bloomery-serve");
        assert_eq!(out.status.code(), Some(0), "{flag}: {out:?}");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            text.starts_with("usage: bloomery-serve"),
            "{flag}: stdout {text:?}"
        );
        assert!(
            out.stderr.is_empty(),
            "{flag}: the usage is no error: {:?}",
            out.stderr
        );
    }
}

/// A refusal before the listener binds exits with its code and prints one `error` event, and
/// nothing else on stdout.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_refusal_exits_with_its_code_and_prints_one_error_event() {
    let absent = std::env::temp_dir().join(format!(
        "bloomery-serve-contract-{}-absent.gguf",
        std::process::id()
    ));
    let absent = absent.to_str().expect("a UTF-8 temp path");
    let cases: [(&[&str], u8, &str); 2] = [
        (&["--no-such-flag"], 64, "usage"),
        (&["--port", "0", "--model", absent], 66, "no-model"),
    ];
    for (args, status, code) in cases {
        let out = Command::new(MOCK)
            .args(args)
            .output()
            .expect("run bloomery-serve");
        assert_eq!(
            out.status.code(),
            Some(i32::from(status)),
            "{args:?}: {out:?}"
        );
        let text = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1, "{args:?}: stdout {text:?}");
        let v: Value = serde_json::from_str(lines[0]).expect("an event line");
        assert_eq!(
            (&v["event"], &v["exit"], &v["code"]),
            (&json!("error"), &json!(status), &json!(code)),
            "{args:?}: {v}"
        );
        for key in ["message", "hint"] {
            assert!(
                v[key].as_str().is_some_and(|s| !s.is_empty()),
                "{args:?}: no {key}: {v}"
            );
        }
    }
}
