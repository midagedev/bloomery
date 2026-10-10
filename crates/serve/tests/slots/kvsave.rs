//! Gate: the prompt cache's snapshot leaves the request path. Over a mock that
//! copies its states on a stream of its own ([`MockEngine::with_async_snapshots`]),
//! a request's end begins the copy of its slot's state: the next conversation
//! on that slot cuts it with no snapshot taken on its path (`cache_ms` far
//! under the copy's time, no snapshot call counted), a request that arrives
//! while the copy still runs waits for it before the rows are cut (no call
//! ever rewrote rows a copy was reading), and the cached conversation comes
//! back as the prefix its request keeps — the positions the engine took back
//! are those, not the state's.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Duration;

use serde_json::Value;
use serve::{MockEngine, SaveStats};

use super::{completion, start_n};
use crate::common::post;

/// The time one state's copy takes on the mock's copy stream.
const COPY: Duration = Duration::from_millis(200);

/// A server of one slot over the async-snapshot mock with room for a few
/// states.
fn server(copy: Duration) -> (SocketAddr, Arc<SaveStats>) {
    let (mock, stats) = MockEngine::new(4096)
        .with_cache_ram(1 << 20)
        .with_async_snapshots(copy);
    (start_n(Box::new(mock), 1, None, None), stats)
}

/// One greedy request: its response.
fn ask(addr: SocketAddr, prompt: &str) -> Value {
    let r = post(addr, "/completion", &completion(prompt, 8));
    assert_eq!(r.status, 200, "{}", r.body);
    r.json()
}

fn cache_ms(v: &Value) -> f64 {
    v["timings"]["cache_ms"].as_f64().expect("cache_ms")
}

/// A second conversation after the first one's copy landed pays no snapshot
/// on its path, and the first one back is served from the cache as the prefix
/// it keeps.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_conversation_switch_pays_no_snapshot_and_a_hit_puts_back_its_prefix() {
    let (addr, stats) = server(COPY);
    let first = "<｜User｜>the cat sat on the mat and the dog sat on the log and the ";
    let a = ask(addr, first);
    std::thread::sleep(COPY * 2);
    let b = ask(addr, "xyz uvw xyz uvw xyz ");
    assert!(
        cache_ms(&b) < COPY.as_secs_f64() * 1e3 / 4.0,
        "cache_ms {} of a switch that waits for nothing",
        cache_ms(&b)
    );
    assert_eq!(
        stats.sync_snapshots.load(SeqCst),
        0,
        "a snapshot on the request path"
    );
    std::thread::sleep(COPY * 2);
    // The first conversation again, its reply appended and a new turn after it.
    let again = format!(
        "{first}{}<｜User｜>and then the ",
        a["content"].as_str().expect("content")
    );
    let r = ask(addr, &again);
    let kept = r["timings"]["cache_n"].as_u64().expect("cache_n");
    assert!(kept > 0, "nothing kept of the first conversation: {r}");
    assert_eq!(
        stats.resumed.load(SeqCst) as u64,
        kept,
        "positions put back are the kept ones"
    );
    assert_eq!(
        stats.whole_resumes.load(SeqCst),
        0,
        "a state went back whole"
    );
    assert_eq!(stats.hazards.load(SeqCst), 0, "a rewrite under a copy");
}

/// A request that arrives while its slot's copy still runs waits for it:
/// `cache_ms` carries the wait, and no cut, reset or resume reached rows the
/// copy was reading.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_request_under_a_running_copy_waits_for_it_and_rewrites_nothing_under_it() {
    let (addr, stats) = server(COPY);
    ask(
        addr,
        "<｜User｜>the cat sat on the mat and the dog sat on the log and the ",
    );
    let b = ask(addr, "xyz uvw xyz uvw xyz ");
    assert_eq!(stats.hazards.load(SeqCst), 0, "a rewrite under a copy");
    assert_eq!(stats.sync_snapshots.load(SeqCst), 0);
    assert!(
        cache_ms(&b) > COPY.as_secs_f64() * 1e3 / 4.0,
        "cache_ms {} of a request that had a copy to wait for",
        cache_ms(&b)
    );
}
