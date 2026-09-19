//! Gate for the KV cache: caching must change **nothing**.
//!
//! Not "within a tolerance" — nothing. The cache stores the same f16 `kvr` rows the
//! uncached path built and threw away each pass, attention walks the same keys in the
//! same order, and `matmul_q` quantizes activations per column so the batch width does
//! not reach the arithmetic. Every one of those is a claim, and a tolerance here would
//! let all three be slightly false at once. So the gate is `max|diff| == 0`, the same
//! shape as the `inp_embd` gate in `tests/forward.rs`.
//!
//! What this catches that a tolerance would not: a mask built against the wrong slot
//! array, a block whose rows fall a position behind the table, a query indexed by the
//! key count. Each of those moves the answer by an amount that looks like drift.
//!
//! `hw_` prefix: needs the box and the model file. It does NOT need the oracle — this
//! compares our two paths against each other, and `tests/forward.rs` is what ties the
//! uncached one to ik.
#[path = "common/oracle.rs"]
mod oracle;

use model::forward::{forward, new_cache, step};

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "length {} vs {}",
        got.len(),
        want.len()
    );
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

/// One `step` on an empty cache must equal the uncached forward exactly. This is the
/// shim's own claim: `block_attn_trace` now runs the cached code against a cache holding
/// only its batch, so any difference here means the shim is not the identity it says
/// it is.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_kv_one_shot_equals_uncached() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();

    let plain = forward(&g, &tokens).unwrap();
    let mut cache = new_cache(&g).unwrap();
    let cached = step(&g, &tokens, &mut cache).unwrap();

    assert_eq!(cache.len(), tokens.len(), "the cache holds the whole batch");
    let worst = max_abs_diff(&cached.data, &plain.data);
    assert_eq!(
        worst, 0.0,
        "one-shot cached vs uncached: max|diff| {worst:e}"
    );
    eprintln!("one-shot cached == uncached          max|diff| = 0   exact");
}

/// Prefill n-1, then one decode step: the logits must equal the n-token one-shot pass
/// exactly. This is the property the whole cache exists for — if it holds, a decode step
/// is free to stop re-reading the prefix.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_kv_incremental_is_bit_exact() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();
    let split = tokens.len() - 1;

    let one_shot = forward(&g, &tokens).unwrap();

    let mut cache = new_cache(&g).unwrap();
    step(&g, &tokens[..split], &mut cache).unwrap();
    assert_eq!(cache.len(), split, "prefill cached {split} positions");
    assert_eq!(
        cache.next_pos(0),
        split as u32,
        "next position follows the table"
    );
    let incremental = step(&g, &tokens[split..], &mut cache).unwrap();
    assert_eq!(cache.len(), tokens.len(), "the step appended its own token");

    let worst = max_abs_diff(&incremental.data, &one_shot.data);
    assert_eq!(
        worst,
        0.0,
        "prefill {split} + 1 step vs one-shot {}: max|diff| {worst:e}",
        tokens.len()
    );
    eprintln!(
        "prefill {split} + 1 step == one-shot {}   max|diff| = 0   exact",
        tokens.len()
    );
}

/// The same equality one token at a time, all the way down. A cache that is right for
/// one step and wrong for the fifth is the failure mode a single split would miss — the
/// slot table and the per-block row counts only drift apart after repetition.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_kv_every_split_is_bit_exact() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();

    // One reference per prefix length, from the uncached path.
    let refs: Vec<Vec<f32>> = (1..=tokens.len())
        .map(|n| forward(&g, &tokens[..n]).unwrap().data)
        .collect();

    let mut cache = new_cache(&g).unwrap();
    for (i, t) in tokens.iter().enumerate() {
        let got = step(&g, &[*t], &mut cache).unwrap();
        let worst = max_abs_diff(&got.data, &refs[i]);
        eprintln!(
            "token {i} (ctx {})                        max|diff| = {worst:e}",
            i + 1
        );
        assert_eq!(worst, 0.0, "step {i}: max|diff| {worst:e}");
    }
    assert_eq!(cache.len(), tokens.len());
}
