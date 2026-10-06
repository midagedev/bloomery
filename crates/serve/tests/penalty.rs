//! Gate: the repetition, frequency and presence penalties as llama-server takes
//! them — its field names and defaults, llama.cpp's checks on the values, and
//! the penalties reaching the sampled ids — over the mock engine.

mod common;

use std::net::SocketAddr;

use common::{Reply, get, post, start};
use serde_json::{Value, json};
use serve::Tokenizer;

const N_CTX: usize = 4096;

/// `base` with `extra`'s keys set over it.
fn with(mut base: Value, extra: &Value) -> Value {
    for (k, v) in extra.as_object().expect("an object") {
        base[k] = v.clone();
    }
    base
}

/// A one-token `/completion` of "ab" with `extra` in the body.
fn short(addr: SocketAddr, extra: &Value) -> Reply {
    post(
        addr,
        "/completion",
        &with(json!({"prompt": "ab", "n_predict": 1}), extra),
    )
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_props_report_the_penalty_defaults() {
    let addr = start(N_CTX);
    let s = get(addr, "/props").json()["default_generation_settings"].clone();
    // llama.cpp `common/common.h:239-242`.
    assert_eq!(s["repeat_last_n"], 64, "{s}");
    assert_eq!(s["repeat_penalty"], 1.0, "{s}");
    assert_eq!(s["frequency_penalty"], 0.0, "{s}");
    assert_eq!(s["presence_penalty"], 0.0, "{s}");
    assert_eq!(s["samplers"][0], "penalties", "{s}");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_penalty_fields_reach_the_settings() {
    let addr = start(N_CTX);
    let settings = |extra: Value| -> Value {
        let r = short(addr, &extra);
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["generation_settings"].clone()
    };
    let s = settings(json!({
        "repeat_penalty": 1.25,
        "frequency_penalty": 0.25,
        "presence_penalty": 1.5,
        "repeat_last_n": 32,
    }));
    assert_eq!(
        (
            &s["repeat_penalty"],
            &s["frequency_penalty"],
            &s["presence_penalty"],
            &s["repeat_last_n"]
        ),
        (&json!(1.25), &json!(0.25), &json!(1.5), &json!(32)),
        "{s}"
    );
    // `null` is absent.
    let s = settings(json!({"presence_penalty": null}));
    assert_eq!(s["presence_penalty"], 0.0, "{s}");
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_bad_penalties_are_named_400s() {
    let addr = start(N_CTX);
    let refused = [
        (json!({"presence_penalty": 1e39}), "presence_penalty"),
        (json!({"frequency_penalty": -1e39}), "frequency_penalty"),
        (json!({"frequency_penalty": "0.5"}), "frequency_penalty"),
        (json!({"repeat_penalty": 0}), "repeat_penalty"),
        (json!({"repeat_penalty": -1.1}), "repeat_penalty"),
        // A subnormal f32: its reciprocal overflows (`common/sampling.cpp:190-193`).
        (json!({"repeat_penalty": 1e-39}), "repeat_penalty"),
        // llama-server takes `0..=i32::MAX` (`tools/server/server-schema.cpp:126`).
        (json!({"repeat_last_n": -1}), "repeat_last_n"),
        (json!({"repeat_last_n": 2_147_483_648_i64}), "repeat_last_n"),
        (json!({"repeat_last_n": 2.5}), "repeat_last_n"),
        (json!({"repeat_last_n": "64"}), "repeat_last_n"),
        (json!({"logit_bias": [[7, 1.0]]}), "logit_bias"),
        (json!({"logit_bias": {"7": false}}), "logit_bias"),
    ];
    for (extra, field) in refused {
        let r = short(addr, &extra);
        assert_eq!(r.status, 400, "{extra}: {}", r.body);
        let e = &r.json()["error"];
        assert_eq!(e["type"], "invalid_request_error", "{extra}: {e}");
        assert!(
            e["message"].as_str().is_some_and(|m| m.contains(field)),
            "{extra}: the message names {field}: {e}"
        );
    }
    let passed = [
        json!({"logit_bias": []}),
        json!({"logit_bias": {}}),
        json!({"temperature": 0, "presence_penalty": 1.5}),
        json!({"temperature": 0, "presence_penalty": 1.5, "repeat_last_n": 0}),
        json!({"temperature": 0, "presence_penalty": 0, "repeat_penalty": 1}),
    ];
    for extra in passed {
        let r = short(addr, &extra);
        assert_eq!(r.status, 200, "{extra}: {}", r.body);
    }
}

#[test]
#[ignore = "gate: just gate-serve"]
fn hw_presence_penalty_changes_the_sampled_ids() {
    let addr = start(N_CTX);
    // top_k 1 makes the draw the best candidate: the mock's echo of "ab".
    let ids = |extra: Value| -> Value {
        let base = json!({
            "prompt": "abababab",
            "n_predict": 6,
            "temperature": 1.0,
            "top_k": 1,
            "seed": 1,
            "return_tokens": true,
        });
        let r = post(addr, "/completion", &with(base, &extra));
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["tokens"].clone()
    };
    let plain = ids(json!({}));
    assert_eq!(
        plain.as_array().map(Vec::len),
        Some(6),
        "the echo runs to n_predict: {plain}"
    );
    assert_eq!(ids(json!({"presence_penalty": 0.0})), plain, "neutral");
    let penalized = ids(json!({"presence_penalty": 20.0}));
    assert_ne!(
        penalized, plain,
        "presence 20 takes the echoed id below the mock's floor"
    );
    // A greedy request takes the argmax after the penalties, as llama-server's
    // chain does: top_k 1's draw.
    assert_eq!(ids(json!({"temperature": 0.0})), plain, "greedy, neutral");
    assert_eq!(
        ids(json!({"temperature": 0.0, "presence_penalty": 20.0})),
        penalized,
        "greedy, presence 20"
    );
    println!("presence: {plain} neutral, {penalized} at 20");
}

/// The penalty window opens on the prompt's ids, as llama-server's
/// `init_sampler` does (it feeds every prompt id into the sampler before the
/// first draw): the mock answers "abababab" with its bigram echo `a` — the
/// follower of the last `b`'s most recent earlier occurrence, the PREDICTED
/// logit — and a presence penalty that takes that `a` below the mock's floor
/// moves the first id only while the window reaches into the prompt. The
/// same request with the penalty off, or with a window too short to hold an
/// `a`, answers the echo; a rerun over the cached prompt answers what the
/// cold one did.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_the_penalty_window_opens_on_the_prompts_ids() {
    let addr = start(N_CTX);
    // The echoed `a`, and the first id at the mock's floor (-8, the lowest
    // id once the echo is pushed under it).
    let (a, floor) = (
        u64::from(serve::MockTokenizer.encode("a")[0]),
        u64::from(serve::MockTokenizer.bos()),
    );
    let first = |extra: Value| -> u64 {
        let mut body = json!({
            "prompt": "abababab",
            "n_predict": 1,
            "temperature": 0,
            "return_tokens": true,
        });
        for (k, v) in extra.as_object().expect("an object") {
            body[k] = v.clone();
        }
        let r = post(addr, "/completion", &body);
        assert_eq!(r.status, 200, "{body}: {}", r.body);
        r.json()["tokens"][0]
            .as_u64()
            .unwrap_or_else(|| panic!("{body}: no first token: {}", r.body))
    };
    assert_eq!(first(json!({})), a, "greedy answers the mock's echo");
    assert_eq!(
        first(json!({"presence_penalty": 0.0})),
        a,
        "the penalty off"
    );
    // The echoed `a` (logit 4) and the prompt's `b` sit in the window: both
    // go under every other id's floor, and the draw moves to the first of
    // those.
    assert_eq!(
        first(json!({"presence_penalty": 20.0})),
        floor,
        "the prompt's ids are in the window"
    );
    // A window of one id holds only the prompt's last id (`b`): the echoed
    // `a` stays unpenalized and wins.
    assert_eq!(
        first(json!({"presence_penalty": 20.0, "repeat_last_n": 1})),
        a,
        "a window short of the echoed id"
    );
    // The rerun's prompt is the slot's cache: the cache keeps positions, not
    // the history the draws read.
    assert_eq!(
        first(json!({"presence_penalty": 20.0})),
        floor,
        "the cached rerun"
    );
}
