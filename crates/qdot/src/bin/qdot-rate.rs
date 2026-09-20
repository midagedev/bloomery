//! Kernel-rate bench for `dot_row` alone (MUL-26 diagnosis): how many
//! weight bytes per second does the Q3_K x Q8_K kernel actually stream,
//! outside the engine, on this machine. The pre-study
//! (`crates/q3k-cpu/RESULTS-r4-lead.md`) measured the same arithmetic at
//! 202.5 GB/s on 32 threads built with `target-cpu=znver3`; this bench
//! exists to compare that number against `qdot` as the engine builds it
//! (plain release, no target flags) — the profile table says the engine's
//! fused path streams ~7 GB/s aggregate, and the suspected cause is the
//! missing `#[target_feature]` on the kernel fn.
//!
//! Synthetic weights (data-independent kernel) sized to the pre-study's
//! `big` shape: 360,448 rows of K=2048 (880 B each, 317 MB total), one
//! quantized activation column, single-threaded — the per-core rate is the
//! number that isolates the codegen question from threading.

use std::time::Instant;

fn main() {
    let k = 2048usize;
    let nb = k / 256;
    let row_bytes = nb * 110;
    let rows: usize = 360_448;

    // xorshift64* filler: any bytes are valid Q3_K codes, and the kernel's
    // time is data-independent. The activation column goes through the real
    // `quantize_col` so its bytes are what the kernel really consumes.
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545F4914F6CDD1D)
    };
    let mut w = vec![0u8; rows * row_bytes];
    for c in w.chunks_exact_mut(8) {
        c.copy_from_slice(&next().to_le_bytes());
    }
    let col: Vec<f32> = (0..k)
        .map(|i| (((i as i64 % 31) as f32) - 15.0) / 16.0)
        .collect();
    let cb = qdot::col_bytes(gguf::GgmlType::Q3_K, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(gguf::GgmlType::Q3_K, &col, &mut acol);

    // Warm-up + timed passes. The accumulator feeds a println so the loop
    // cannot be optimized away.
    let mut acc = 0.0f32;
    for r in 0..rows.min(4096) {
        acc += qdot::dot_row(gguf::GgmlType::Q3_K, &w[r * row_bytes..], &acol, k).unwrap();
    }
    let passes = 6u32;
    let t0 = Instant::now();
    for _ in 0..passes {
        for r in 0..rows {
            acc += qdot::dot_row(gguf::GgmlType::Q3_K, &w[r * row_bytes..], &acol, k).unwrap();
        }
    }
    let dt = t0.elapsed();
    let bytes = rows as f64 * row_bytes as f64 * passes as f64;
    println!(
        "rows {rows} x {row_bytes} B x {passes} passes in {:.3?} = {:.1} GB/s (sum {acc:.3})",
        dt,
        bytes / dt.as_secs_f64() / 1e9
    );
}
