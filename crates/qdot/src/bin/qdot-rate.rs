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
    for c in w.as_chunks_mut::<8>().0 {
        c.copy_from_slice(&next().to_le_bytes());
    }
    // Every MXFP4 block's E8M0 byte into 112..127, as mxfp4_x4_rate.cpp does: raw bytes
    // give e = 0 or 1, a subnormal f32 scale, in 2 blocks of 256 on average.
    if ty == GgmlType::MXFP4 {
        for b in w.as_chunks_mut::<17>().0 {
            b[0] = 0x70 | (b[0] & 0x0f);
        }
    }
    // IQ4_NL and IQ4_XS carry an f16 d per block: mask its exponent's top bit (the
    // block's second byte) so every scale stays finite, as iq4nl_rate.cpp and
    // iq4xs_rate.cpp do — raw bytes give an infinity or NaN scale in 2 blocks of 256
    // on average.
    if ty == GgmlType::IQ4_NL {
        for b in w.as_chunks_mut::<18>().0 {
            b[1] &= 0x7b;
        }
    }
    if ty == GgmlType::IQ4_XS {
        for b in w.as_chunks_mut::<136>().0 {
            b[1] &= 0x7b;
        }
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
    // Q8_0 at the Q5_1 row's k, so the two compare per byte: 342 x 34 B = 11628 B/row.
    bench(GgmlType::Q8_0, 10944, (10944 / 32) * 34, rows);
    // V4.1 ffn_down_exps shape (layers 0 and 1): k = 2304 (9 x 176 B = 1584 B/row).
    bench(GgmlType::Q5_K, 2304, (2304 / 256) * 176, rows);
    // V4-Flash ffn_gate/up_exps shape: k = 4096 (16 x 98 B = 1568 B/row).
    bench(GgmlType::IQ3_XXS, 4096, (4096 / 256) * 98, rows);
    // V4-Flash and MiMo-V2.6-Flash ffn_down_exps shape: k = 2048 (64 x 17 B = 1088 B/row).
    bench(GgmlType::MXFP4, 2048, (2048 / 32) * 17, rows);
    // MiMo-V2.6-Flash ffn_gate/up_exps shape: k = 4096 (128 x 17 B = 2176 B/row).
    bench(GgmlType::MXFP4, 4096, (4096 / 32) * 17, rows);
    // Qwen3.8 UD-Q3_K_XL ffn_down_exps shape (43 of 48 layers): k = 640 (20 x 18 B =
    // 360 B/row), beside the Q5_1 row it replaces at the same k.
    bench(GgmlType::IQ4_NL, 640, (640 / 32) * 18, rows);
    bench(GgmlType::Q5_1, 640, (640 / 32) * 24, rows);
    // Qwen3.8 UD-Q3_K_XL ffn_gate/up_exps shape (the one IQ4_XS layer): k = 2560
    // (10 x 136 B = 1360 B/row), beside the Q4_K row it replaces at the same k.
    bench(GgmlType::IQ4_XS, 2560, (2560 / 256) * 136, rows);
    bench(GgmlType::Q4_K, 2560, (2560 / 256) * 144, rows);
}
