//! Kernel-rate benchmark for single-threaded `dot_row` streaming bandwidth.

use std::time::Instant;

use gguf::GgmlType;

fn bench(ty: GgmlType, k: usize, row_bytes: usize, rows: usize) {
    // xorshift64* filler: any bytes are valid quantized codes.
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

    // Warm-up + timed passes; accumulate into `acc` to prevent dead-code elimination.
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
    // Row bytes: Q3_K 110 B/256, Q4_K 144 B/256, Q6_K 210 B/256.
    bench(GgmlType::Q3_K, 2048, (2048 / 256) * 110, rows);
    bench(GgmlType::Q4_K, 2048, (2048 / 256) * 144, rows);
    bench(GgmlType::Q6_K, 2048, (2048 / 256) * 210, rows);
    // Real ffn_down_exps shape: k = 1408 (44 x 22 B = 968 B/row).
    bench(GgmlType::Q5_0, 1408, (1408 / 32) * 22, rows);
    // Real ffn_down shape: k = 10944 (342 x 24 B = 8208 B/row).
    bench(GgmlType::Q5_1, 10944, (10944 / 32) * 24, rows);
}
