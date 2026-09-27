//! Gate: what one client cannot do to the server over the mock engine — hold
//! connections past [`serve::MAX_CONNECTIONS`], keep an idle one past
//! [`serve::KEEP_ALIVE_IDLE`], or send a request that takes the engine past the
//! positions it serves.

mod common;

use common::{get, post, start};
use serde_json::json;

/// The connection clauses count the server's threads through `/proc`, so they
/// run on Linux only (the box); the context clause runs everywhere.
#[cfg(target_os = "linux")]
mod connections {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream};
    use std::time::{Duration, Instant};

    use super::common::{get, start};
    use serve::{KEEP_ALIVE_IDLE, MAX_CONNECTIONS};

    /// The server's connection threads (named `serve:<port>`) alive in this process.
    fn connection_threads(port: u16) -> usize {
        let name = format!("serve:{port}");
        std::fs::read_dir("/proc/self/task")
            .unwrap_or_else(|e| panic!("/proc/self/task: {e}"))
            .filter_map(|t| std::fs::read_to_string(t.ok()?.path().join("comm")).ok())
            .filter(|comm| comm.trim_end() == name)
            .count()
    }

    /// Waits until `connection_threads(port)` is `n`; panics with the count after ten seconds.
    fn await_threads(port: u16, n: usize) {
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            let live = connection_threads(port);
            if live == n {
                return;
            }
            assert!(
                Instant::now() < until,
                "{live} connection threads, waiting for {n}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A connection that has sent the first line of a request and waits inside it.
    fn half_request(addr: SocketAddr) -> TcpStream {
        let mut s = TcpStream::connect(addr).expect("connect");
        s.write_all(b"GET /health HTTP/1.1\r\n")
            .expect("write a request line");
        s
    }

    /// `MAX_CONNECTIONS` connections inside a request hold every place: the next one
    /// gets a 503 with `Retry-After` and no thread, and once one closes a new one is
    /// served.
    #[test]
    #[ignore = "gate: just gate-serve"]
    fn hw_connections_past_the_limit_get_503() {
        let addr = start(4096);
        let mut held: Vec<TcpStream> = (0..MAX_CONNECTIONS).map(|_| half_request(addr)).collect();
        await_threads(addr.port(), MAX_CONNECTIONS);

        let refused = get(addr, "/health");
        assert_eq!(refused.status, 503, "{}", refused.body);
        assert_eq!(refused.header("Retry-After"), Some("1"));
        let e = &refused.json()["error"];
        assert_eq!(e["type"], "unavailable_error", "{e}");
        assert!(
            e["message"]
                .as_str()
                .is_some_and(|m| m.contains(&MAX_CONNECTIONS.to_string())),
            "{e}"
        );
        assert_eq!(connection_threads(addr.port()), MAX_CONNECTIONS);

        drop(held.pop());
        await_threads(addr.port(), MAX_CONNECTIONS - 1);
        let served = get(addr, "/health");
        assert_eq!(served.status, 200, "{}", served.body);
        drop(held);
    }

    /// Reads one response with a `Content-Length` body off a keep-alive connection.
    fn read_response(s: &mut TcpStream) -> String {
        let mut raw = Vec::new();
        let mut byte = [0u8; 1];
        while !raw.ends_with(b"\r\n\r\n") {
            let n = s.read(&mut byte).expect("read a response head");
            assert_eq!(n, 1, "the connection closed inside a response head");
            raw.push(byte[0]);
        }
        let head = String::from_utf8(raw).expect("a UTF-8 head");
        let len: usize = head
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: "))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| panic!("no Content-Length in {head:?}"));
        let mut body = vec![0u8; len];
        s.read_exact(&mut body).expect("read a response body");
        head + &String::from_utf8_lossy(&body)
    }

    /// Waits for the server to close `s` and returns how long after `since` it did;
    /// panics if it sends anything or stays open for three idle timeouts.
    fn closed_after(s: &mut TcpStream, since: Instant) -> Duration {
        s.set_read_timeout(Some(KEEP_ALIVE_IDLE * 3))
            .expect("read timeout");
        let mut buf = [0u8; 64];
        match s.read(&mut buf) {
            Ok(0) => since.elapsed(),
            Ok(n) => panic!("an idle connection got {n} bytes"),
            Err(e) => panic!("an idle connection stayed open {:?} ({e})", since.elapsed()),
        }
    }

    /// A connection that sends nothing, and one that goes quiet after a keep-alive
    /// request, are each closed [`KEEP_ALIVE_IDLE`] after they went quiet, and their
    /// threads, with their places, are gone.
    #[test]
    #[ignore = "gate: just gate-serve"]
    fn hw_idle_connections_are_closed() {
        let addr = start(4096);
        // Each clock starts before the server's own wait can: before the connect,
        // before the request whose answer ends the keep-alive connection's last read.
        let silent_since = Instant::now();
        let mut silent = TcpStream::connect(addr).expect("connect");
        let mut kept = TcpStream::connect(addr).expect("connect");
        let kept_since = Instant::now();
        kept.write_all(b"GET /health HTTP/1.1\r\nHost: gate\r\n\r\n")
            .expect("write a request");
        let answer = read_response(&mut kept);
        assert!(answer.contains("Connection: keep-alive"), "{answer}");
        await_threads(addr.port(), 2);

        for (what, s, since) in [
            ("silent", &mut silent, silent_since),
            ("keep-alive", &mut kept, kept_since),
        ] {
            let after = closed_after(s, since);
            assert!(
                after >= KEEP_ALIVE_IDLE && after < KEEP_ALIVE_IDLE * 2,
                "the {what} connection closed {after:?} after it went quiet"
            );
        }
        await_threads(addr.port(), 0);
    }
}

/// With the mock serving `CTX` positions (it fails a step past them, as V4.1's
/// body does, and the server would end): a prompt of `CTX` + 1 tokens is a 400
/// naming both numbers; a `max_tokens` past the positions is served up to them,
/// `truncated`, as llama-server's slots stop at `n_ctx`; and the server serves
/// the next request.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_requests_stop_at_the_context() {
    const CTX: usize = 64;
    const N: usize = 10;
    let addr = start(CTX);
    // "abcabcabca" as mock byte ids (6 + byte): greedy continues the cycle, no EOS.
    let cycle: Vec<u32> = "abc".bytes().map(|b| 6 + u32::from(b)).collect();
    let ids: Vec<u32> = cycle.iter().copied().cycle().take(N).collect();

    // A prompt of CTX tokens leaves no position for a generated token to be fed at.
    for len in [CTX + 1, CTX] {
        let long: Vec<u32> = cycle.iter().copied().cycle().take(len).collect();
        let refused = post(
            addr,
            "/completion",
            &json!({"prompt": long, "n_predict": 4, "temperature": 0}),
        );
        assert_eq!(refused.status, 400, "a prompt of {len}: {}", refused.body);
        let e = &refused.json()["error"];
        assert_eq!(e["type"], "exceed_context_size_error", "{e}");
        let m = e["message"].as_str().unwrap_or_default();
        for n in [len, CTX] {
            assert!(m.contains(&n.to_string()), "{n} is not in {m:?}");
        }
    }
    // One position short of it is served: the first token, and one fed at CTX - 1.
    let edge: Vec<u32> = cycle.iter().copied().cycle().take(CTX - 1).collect();
    let served = post(
        addr,
        "/completion",
        &json!({"prompt": edge, "n_predict": 4, "temperature": 0}),
    );
    assert_eq!(served.status, 200, "{}", served.body);
    let v = served.json();
    assert_eq!(v["tokens_predicted"], 2, "{v}");
    assert_eq!(v["truncated"], true, "{v}");

    let past = post(
        addr,
        "/completion",
        &json!({"prompt": ids, "max_tokens": CTX, "temperature": 0}),
    );
    assert_eq!(past.status, 200, "{}", past.body);
    let v = past.json();
    // Feeding the k-th generated token takes position N + k - 1: the last
    // position the engine evaluates is CTX - 1, the last token is never fed.
    assert_eq!(v["tokens_predicted"], CTX - N + 1, "{v}");
    assert_eq!(v["truncated"], true, "{v}");
    assert_eq!(v["timings"]["n_past"], CTX, "{v}");

    let next = post(
        addr,
        "/completion",
        &json!({"prompt": ids, "n_predict": 3, "temperature": 0}),
    );
    assert_eq!(next.status, 200, "{}", next.body);
    assert_eq!(next.json()["content"], "bca");
    let health = get(addr, "/health");
    assert_eq!(health.status, 200, "{}", health.body);
    assert_eq!(health.json()["status"], "ok");
}
