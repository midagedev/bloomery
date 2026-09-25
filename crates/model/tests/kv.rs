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
#[path = "common/manifest.rs"]
mod manifest;
#[path = "common/model_path.rs"]
mod model_path;
#[path = "common/prompt.rs"]
mod prompt;

use model::arch::deepseek2::derived::Derived;
use model::arch::deepseek2::forward::{forward, new_cache, step};

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
/// it is. The same run is `Derived`'s end-to-end claim (`tests/derived.rs`, layer 3):
/// `forward` builds its own `Derived`, `step` is handed one, and block `b`'s absorption
/// must get block `b`'s blocks either way. Bits, not differences: `NaN - NaN` folds to
/// a clean zero through `f32::max`.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_kv_one_shot_equals_uncached() {
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let tokens: Vec<u32> = prompt::tokens();

    let plain = forward(&g, &tokens).unwrap();
    let mut cache = new_cache(&g).unwrap();
    let derived = Derived::new(&g).unwrap();
    let cached = step(&g, &tokens, &mut cache, &derived).unwrap();

    assert_eq!(cache.len(), tokens.len(), "the cache holds the whole batch");
    assert_eq!(cached.data.len(), plain.data.len(), "logit count");
    let differing = cached
        .data
        .iter()
        .zip(&plain.data)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let worst = max_abs_diff(&cached.data, &plain.data);
    assert_eq!(
        differing, 0,
        "one-shot cached vs uncached: {differing} logits differ in bits, max|diff| {worst:e}"
    );
    eprintln!("one-shot cached == uncached          bit-identical");
}

/// Prefill n-1, then one decode step: the logits must equal the n-token one-shot pass
/// exactly. This is the property the whole cache exists for — if it holds, a decode step
/// is free to stop re-reading the prefix.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_kv_incremental_is_bit_exact() {
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let tokens: Vec<u32> = prompt::tokens();
    let split = tokens.len() - 1;

    let one_shot = forward(&g, &tokens).unwrap();

    let mut cache = new_cache(&g).unwrap();
    let derived = Derived::new(&g).unwrap();
    step(&g, &tokens[..split], &mut cache, &derived).unwrap();
    assert_eq!(cache.len(), split, "prefill cached {split} positions");
    assert_eq!(
        cache.next_pos(0),
        split as u32,
        "next position follows the table"
    );
    let incremental = step(&g, &tokens[split..], &mut cache, &derived).unwrap();
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
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let tokens: Vec<u32> = prompt::tokens();

    // One reference per prefix length, from the uncached path.
    let refs: Vec<Vec<f32>> = (1..=tokens.len())
        .map(|n| forward(&g, &tokens[..n]).unwrap().into_data())
        .collect();

    let mut cache = new_cache(&g).unwrap();
    let derived = Derived::new(&g).unwrap();
    for (i, t) in tokens.iter().enumerate() {
        let got = step(&g, &[*t], &mut cache, &derived).unwrap();
        let worst = max_abs_diff(&got.data, &refs[i]);
        eprintln!(
            "token {i} (ctx {})                        max|diff| = {worst:e}",
            i + 1
        );
        assert_eq!(worst, 0.0, "step {i}: max|diff| {worst:e}");
    }
    assert_eq!(cache.len(), tokens.len());
}

/// The contiguous storage contract: `keys(b)` views ONE flat row-major buffer —
/// `row(i)` is exactly the i-th `width`-wide slice, and the buffer is `len * width`
/// long. This is the shape the reference's `kv_cache-N` has; anything between `push`
/// and the flash kernel that reinterprets a row (a stale length, a shifted stride)
/// reds this gate before it can hand a kernel another row's bytes. Pure construction
/// — no model file, no oracle.
#[test]
#[ignore = "hw: construction-only; runs with the other kv gates"]
fn hw_kv_rows_view_is_row_major() {
    let width = 12usize;
    let mut cache = model::kv::KvCache::new(2, width);
    let batches: [&[u32]; 2] = [&[7, 11], &[13]];
    // The expected rows in push order — batch, then token, each the f16 of its
    // source column. Distinct values per (batch, token, element) so a shifted
    // row read cannot pass by accident.
    let mut want: Vec<Vec<u16>> = Vec::new();
    for (bi, positions) in batches.iter().enumerate() {
        let slots: Vec<model::Slot> = positions
            .iter()
            .map(|&pos| model::Slot { seq: 0, pos })
            .collect();
        let range = cache.begin(&slots);
        let mut flat: Vec<u16> = Vec::with_capacity(slots.len() * width);
        for t in 0..slots.len() {
            let col: Vec<u16> = (0..width)
                .map(|e| gguf::quant::f32_to_f16_bits((bi * 100 + t * 10 + e) as f32))
                .collect();
            flat.extend_from_slice(&col);
            want.push(col);
        }
        for b in 0..2 {
            cache.push(b, &range, &flat);
        }
    }
    for b in 0..2 {
        let keys = cache.keys(b);
        assert_eq!(keys.len(), want.len(), "block {b} row count");
        assert_eq!(keys.width(), width, "block {b} row width");
        assert_eq!(
            keys.as_slice().len(),
            want.len() * width,
            "block {b}: one flat buffer of len * width"
        );
        for (i, w) in want.iter().enumerate() {
            assert_eq!(keys.row(i), &w[..], "block {b} row {i}");
        }
    }
}
