//! Kernel-rate bench for `dot_row` alone (MUL-26 diagnosis): how many
//! weight bytes per second does the fused kernel actually stream, outside
//! the engine, on this machine. The pre-study (`crates/q3k-cpu/RESULTS-r4-lead.md`)
//! measured the same Q3_K arithmetic at 202.5 GB/s on 32 threads built with
//! `target-cpu=znver3`; this bench exists to compare that number against
//! `qdot` as the engine builds it (plain release, no target flags).
//!
//! MUL-27 (x4 round): a second bench for Q4_K x Q8_2_X4, the pairing the
//! oracle dispatches — same shape discipline, weight rows sized the way the
//! model stores them (144 B per 256 values). The Q4_K number is the one the
//! round's "better than the port" mandate is judged by: ik's measured
//! reference rate is in the rig-log record of this round, not invented here.
//!
//! Synthetic weights (data-independent kernel) sized to the pre-study's
//! `big` shape: 360,448 rows of K=2048, one quantized activation column,
//! single-threaded — the per-core rate is the number that isolates the
//! codegen question from threading.

use std::time::Instant;

use gguf::GgmlType;

fn bench(ty: GgmlType, k: usize, row_bytes: usize, rows: usize) {
    // xorshift64* filler: any bytes are valid quantized codes, and the
    // kernel's time is data-independent. The activation column goes through
    // the real `quantize_col` so its bytes are what the kernel really
    // consumes.
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
    let cb = qdot::col_bytes(ty, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(ty, &col, &mut acol);

    // Warm-up + timed passes. The accumulator feeds a println so the loop
    // cannot be optimized away.
    let mut acc = 0.0f32;
    for r in 0..rows.min(4096) {
        acc += qdot::dot_row(ty, &w[r * row_bytes..], &acol, k).unwrap();
    }
    let passes = 6u32;
    let t0 = Instant::now();
    for _ in 0..passes {
        for r in 0..rows {
            acc += qdot::dot_row(ty, &w[r * row_bytes..], &acol, k).unwrap();
        }
    }
    let dt = t0.elapsed();
    let bytes = rows as f64 * row_bytes as f64 * passes as f64;
    println!(
        "{ty:?}  k={k} rows {rows} x {row_bytes} B x {passes} passes in {:.3?} = {:.1} GB/s (sum {acc:.3})",
        dt,
        bytes / dt.as_secs_f64() / 1e9
    );
}

fn main() {
    let rows: usize = 360_448;
    // Q3_K: 110 B per 256 values; Q4_K: 144 B per 256 values. MUL-32 adds
    // Q5_0 at the REAL site's shape: ffn_down_exps rows are k = 1408
    // (44 x 22 B = 968 B/row) — the stage table's largest site, and the
    // shape the engine will actually feed this kernel.
    bench(GgmlType::Q3_K, 2048, (2048 / 256) * 110, rows);
    bench(GgmlType::Q4_K, 2048, (2048 / 256) * 144, rows);
    bench(GgmlType::Q5_0, 1408, (1408 / 32) * 22, rows);
}
