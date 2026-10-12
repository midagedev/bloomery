//! Gates: the server drives each request's residency stream through the
//! engine's warm calls ([`Engine::warm_close`], [`Engine::warm_open`]) — how
//! the prompt cache served a start, the slot a call reaches, the expected
//! tokens, a parked request's section, and one close a request.

use std::io::Read;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::testserve::{mock_config, roundtrip, send_request};
use crate::Server;
use crate::engine::{Engine, WarmWhy};
use crate::mock::{MockEngine, WarmCall};
use crate::sched::SlotConfig;
use crate::swap::{Park, SwapEngine};

type Log = Arc<Mutex<Vec<WarmCall>>>;

fn start(engine: Box<dyn Engine>, parallel: usize) -> std::net::SocketAddr {
    let slots = SlotConfig {
        parallel,
        ..SlotConfig::default()
    };
    let server = Server::bind_with("127.0.0.1:0", engine, mock_config(), slots).expect("bind");
    server.spawn().expect("spawn")
}

/// The answer of one completion of `n_predict` tokens at most.
fn complete(addr: std::net::SocketAddr, prompt: &str, n_predict: i64) -> Value {
    let body = json!({"prompt": prompt, "n_predict": n_predict, "temperature": 0});
    let (status, body) = roundtrip(addr, "POST", "/completion", &[], &body.to_string());
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).expect("the answer is JSON")
}

/// Completions of `n_predict` tokens at most for `prompts`, sent one after
/// the other before any answer is read, so they run together.
fn complete_together(addr: std::net::SocketAddr, prompts: [&str; 2], n_predict: i64) {
    let sent: Vec<_> = prompts
        .iter()
        .map(|p| {
            let body = json!({"prompt": p, "n_predict": n_predict, "temperature": 0});
            send_request(addr, "POST", "/completion", &[], &body.to_string())
        })
        .collect();
    for mut stream in sent {
        let mut answer = String::new();
        stream.read_to_string(&mut answer).expect("an answer");
        assert!(answer.starts_with("HTTP/1.0 200"), "{answer}");
    }
}

fn calls(log: &Log) -> Vec<WarmCall> {
    log.lock().expect("the warm log").clone()
}

fn opens(log: &Log) -> Vec<(Option<Vec<u8>>, f64, WarmWhy)> {
    calls(log)
        .into_iter()
        .filter_map(|c| match c {
            WarmCall::Open {
                section,
                expected,
                why,
                ..
            } => Some((section, expected, why)),
            WarmCall::Close { .. } => None,
        })
        .collect()
}

/// A request, an unrelated one that moves its state into the prompt cache,
/// then a request that shares the first one's prompt: the log of the three.
fn served_in_three(engine: MockEngine) -> Vec<WarmCall> {
    let (engine, log) = engine.with_warm();
    let addr = start(Box::new(engine), 1);
    complete(addr, "abcabcab", 4);
    complete(addr, "cbacbacb", 4);
    complete(addr, "abcabcabca", 4);
    calls(&log)
}

/// A full hit hands the entry's section to the stream (`Hit`), an entry with
/// none opens it fresh (`NoSection`), a cut hit and a miss open it fresh.
#[test]
fn the_start_names_how_the_cache_served_it() {
    let cached = || MockEngine::new(4096).with_cache_ram(1 << 20);
    let section_of_first = Some(b"section 0 of slot 0".to_vec());
    let open_of = |c: &WarmCall| match c {
        WarmCall::Open { section, why, .. } => Some((section.clone(), *why)),
        WarmCall::Close { .. } => None,
    };
    let third = |log: Vec<WarmCall>| {
        let opens: Vec<_> = log.iter().filter_map(open_of).collect();
        assert_eq!(opens.len(), 3, "{log:?}");
        assert_eq!(opens[0], (None, WarmWhy::Miss));
        assert_eq!(opens[1], (None, WarmWhy::Miss));
        opens[2].clone()
    };
    assert_eq!(
        third(served_in_three(cached())),
        (section_of_first, WarmWhy::Hit)
    );
    assert_eq!(
        third(served_in_three(cached().with_warm_gap(0))),
        (None, WarmWhy::NoSection)
    );
    assert_eq!(
        third(served_in_three(cached().with_keep_step(2))),
        (None, WarmWhy::Cut)
    );
}

/// With two slots decoding at once, each call reaches the slot of its own
/// request: every slot's stream opens and closes in turn.
#[test]
fn each_request_reaches_the_engine_on_its_own_slot() {
    let (engine, log) = MockEngine::new(4096).with_slots(2).with_warm();
    let addr = start(Box::new(engine), 2);
    complete_together(addr, ["abcabcab", "cbacbacb"], 300);
    let log = calls(&log);
    let mut seen = [Vec::new(), Vec::new()];
    for c in &log {
        match c {
            WarmCall::Open { slot, .. } => seen[*slot].push("open"),
            WarmCall::Close { slot, .. } => seen[*slot].push("close"),
        }
    }
    assert_eq!(
        seen,
        [vec!["open", "close"], vec!["open", "close"]],
        "{log:?}"
    );
}

/// A parked request's section returns to it at hand-back (`Resume`), and the
/// request that used the slot meanwhile closes and opens its own.
#[test]
fn a_parked_request_gets_its_own_section_back() {
    let (engine, log) = MockEngine::new(4096).with_cache_ram(1 << 20).with_warm();
    let engine = SwapEngine::new(Box::new(engine), 2, Park::States { budget: 1 << 20 })
        .expect("a swap engine");
    let addr = start(Box::new(engine), 2);
    complete_together(addr, ["abcabcab", "cbacbacb"], 600);
    let log = calls(&log);
    let closed: Vec<Vec<u8>> = log
        .iter()
        .filter_map(|c| match c {
            WarmCall::Close { section, .. } => section.clone(),
            WarmCall::Open { .. } => None,
        })
        .collect();
    let resumes: Vec<_> = opens(&Log::new(Mutex::new(log.clone())))
        .into_iter()
        .filter(|o| o.2 == WarmWhy::Resume)
        .collect();
    assert!(!resumes.is_empty(), "the requests took turns: {log:?}");
    for (section, _, _) in &resumes {
        let section = section
            .as_ref()
            .expect("a resume carries the parked section");
        assert!(closed.contains(section), "{log:?}");
    }
    // Both requests opened fresh once, each stream closed once per park and
    // once at its request's end.
    let fresh = opens(&Log::new(Mutex::new(log.clone())))
        .into_iter()
        .filter(|o| o.2 == WarmWhy::Miss)
        .count();
    assert_eq!(fresh, 2, "{log:?}");
    assert_eq!(closed.len(), resumes.len() + 2, "{log:?}");
}

/// Every request's E is 1 before the first completion; after it, the budget
/// bounded by the completed requests' mean length (the mean alone when the
/// budget is unbounded), never below 1.
#[test]
fn the_expected_tokens_are_the_budget_bounded_by_the_mean_length() {
    let (engine, log) = MockEngine::new(4096).with_warm();
    let addr = start(Box::new(engine), 1);
    let first = complete(addr, "abcabcab", 5)["tokens_predicted"]
        .as_u64()
        .expect("count");
    let second = complete(addr, "cbacbacb", 100)["tokens_predicted"]
        .as_u64()
        .expect("count");
    complete(addr, "bcabcabc", -1);
    complete(addr, "acbacbac", 0);
    let mean = (first + second) as f64 / 2.0;
    let e: Vec<f64> = opens(&log).into_iter().map(|o| o.1).collect();
    assert_eq!(e, [1.0, (first as f64).clamp(1.0, 5.0), mean, 1.0], "{e:?}");
}

/// A request's stream closes once, at its end, whether or not its state is
/// cached: the opens and closes alternate, and the sync save of the state
/// that follows closes nothing.
#[test]
fn each_requests_stream_closes_exactly_once() {
    let log = served_in_three(MockEngine::new(4096).with_cache_ram(1 << 20));
    let kinds: Vec<bool> = log
        .iter()
        .map(|c| matches!(c, WarmCall::Open { .. }))
        .collect();
    assert_eq!(kinds, [true, false, true, false, true, false], "{log:?}");
}
