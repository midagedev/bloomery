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
    // IQ3_S carries an f16 d per block as well (its first two bytes): the same mask, as
    // iq3s_rate.cpp does — raw bytes give an infinity or NaN scale in one block of 32.
    if ty == GgmlType::IQ3_S {
        for b in w.as_chunks_mut::<110>().0 {
            b[1] &= 0x7b;
        }
    }
    // Q2_K carries two f16 fields per block (d at byte 80, dmin at byte 82) and IQ2_XS one (d at
    // byte 0): the same mask on each field's high byte, as q2k_rate.cpp and iq2xs_rate.cpp do —
    // raw bytes give an infinity or NaN in one Q2_K block of 16 and one IQ2_XS block of 32.
    if ty == GgmlType::Q2_K {
        for b in w.as_chunks_mut::<84>().0 {
            b[81] &= 0x7b;
            b[83] &= 0x7b;
        }
    }
    if ty == GgmlType::IQ2_XS {
        for b in w.as_chunks_mut::<74>().0 {
            b[1] &= 0x7b;
        }
    }
    // NVFP4 rows carry a UE4M3 scale byte at the head of each 9-byte group: raw bytes give a
    // non-scale (bit 7 set, or 0x7F) in 129 groups of 256 on average, which makes the whole row
    // NaN. Every scale byte into 0x30..0x3f.
    if ty == GgmlType::NVFP4_ROWS {
        for g in w.as_chunks_mut::<9>().0 {
            g[0] = 0x30 | (g[0] & 0x0f);
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

/// The tile kernel's rate: `rows` weight rows each dotted against `cols` columns through
/// `dot_row_cols`, single-threaded, as the host union walks a run of columns. Prints the
/// weight bytes streamed per second and the time per (row, column) dot, the number
/// `dot_row`'s line gives as bytes over time for one column. MXFP4 and Q6_K only, the types
/// whose filler is written: MXFP4's E8M0 byte in its `bench` range, and Q6_K's f16 `d` (bytes
/// 208..210 of the 210 B block, after ql[128], qh[64] and the 16 i8 scales) at a fixed normal
/// exponent (2^-5) with two random mantissa bits, since random bytes make it a NaN or an
/// infinity in 2 blocks of 64 on average. Any other type is refused by name.
fn bench_tile(ty: GgmlType, k: usize, row_bytes: usize, rows: usize, cols: usize) {
    assert!(
        matches!(ty, GgmlType::MXFP4 | GgmlType::Q6_K),
        "bench_tile fills MXFP4 and Q6_K blocks only, got {ty:?}"
    );
    assert!(qdot::has_tile(ty), "{ty:?} has no tile kernel on this CPU");
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
    if ty == GgmlType::MXFP4 {
        for b in w.as_chunks_mut::<17>().0 {
            b[0] = 0x70 | (b[0] & 0x0f);
        }
    } else {
        for b in w.as_chunks_mut::<210>().0 {
            b[209] = 0x28 | (b[209] & 0x03);
        }
    }
    let cb = qdot::col_bytes(ty, k);
    let acols: Vec<Vec<u8>> = (0..cols)
        .map(|j| {
            let col: Vec<f32> = (0..k)
                .map(|i| ((((i + 7 * j) as i64 % 31) as f32) - 15.0) / 16.0)
                .collect();
            let mut acol = vec![0u8; cb];
            qdot::quantize_col(ty, &col, &mut acol);
            acol
        })
        .collect();
    let refs: Vec<&[u8]> = acols.iter().map(Vec::as_slice).collect();
    let mut out = vec![0.0f32; cols];

    let mut acc = 0.0f32;
    for r in 0..rows.min(4096) {
        qdot::dot_row_cols(ty, &w[r * row_bytes..], &refs, k, &mut out).unwrap();
        acc += out.iter().sum::<f32>();
    }
    let passes = 6u32;
    let t0 = Instant::now();
    for _ in 0..passes {
        for r in 0..rows {
            qdot::dot_row_cols(ty, &w[r * row_bytes..], &refs, k, &mut out).unwrap();
            acc += out.iter().sum::<f32>();
        }
    }
    let dt = t0.elapsed();
    let bytes = rows as f64 * row_bytes as f64 * passes as f64;
    let dots = rows as f64 * cols as f64 * passes as f64;
    println!(
        "{ty:?} tile C={cols}  k={k} rows {rows} x {row_bytes} B x {passes} passes in {:.3?} = {:.1} GB/s weights, {:.2} ns per row-column (sum {acc:.3})",
        dt,
        bytes / dt.as_secs_f64() / 1e9,
        dt.as_secs_f64() * 1e9 / dots
    );
}

/// The NVFP4 plane tile's rate: `rows` plane rows (`k / 2` code bytes and `k / 16` scale bytes each) dotted
/// against `cols` columns through `dot_nvfp4_planes_rows_cols`, one call a pass over every row, single-threaded, as
/// the host union walks a pair's rows. One column runs the one-column kernel, so the tile's ratio reads against
/// C = 1 on the same filler.
/// The codes are any bytes and each scale a UE4M3 byte in 0x30..0x50, a finite window.
fn bench_planes_tile(k: usize, rows: usize, cols: usize) {
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545F4914F6CDD1D)
    };
    let (code_row, scale_row) = (k / 2, k / 16);
    let mut codes = vec![0u8; rows * code_row];
    for c in codes.as_chunks_mut::<8>().0 {
        c.copy_from_slice(&next().to_le_bytes());
    }
    let scales: Vec<u8> = (0..rows * scale_row)
        .map(|_| 0x30 + (next() % 0x20) as u8)
        .collect();
    let ty = GgmlType::NVFP4_ROWS;
    let cb = qdot::col_bytes(ty, k);
    let acols: Vec<Vec<u8>> = (0..cols)
        .map(|j| {
            let col: Vec<f32> = (0..k)
                .map(|i| ((((i + 7 * j) as i64 % 31) as f32) - 15.0) / 16.0)
                .collect();
            let mut acol = vec![0u8; cb];
            qdot::quantize_col(ty, &col, &mut acol);
            acol
        })
        .collect();
    let refs: Vec<&[u8]> = acols.iter().map(Vec::as_slice).collect();
    let mut acc = 0.0f32;
    let mut pass = |rows: std::ops::Range<usize>| {
        qdot::dot_nvfp4_planes_rows_cols(
            (&codes, &scales),
            rows,
            k,
            (cols, |j| refs[j]),
            |_, _, v| acc += v,
        );
    };
    pass(0..rows.min(4096));
    let passes = 6u32;
    let t0 = Instant::now();
    for _ in 0..passes {
        pass(0..rows);
    }
    let dt = t0.elapsed();
    let row_bytes = code_row + scale_row;
    let bytes = rows as f64 * row_bytes as f64 * passes as f64;
    let dots = rows as f64 * cols as f64 * passes as f64;
    println!(
        "NVFP4_ROWS planes tile C={cols}  k={k} rows {rows} x {row_bytes} B x {passes} passes in {:.3?} = {:.1} GB/s weights, {:.2} ns per row-column (sum {acc:.3})",
        dt,
        bytes / dt.as_secs_f64() / 1e9,
        dt.as_secs_f64() * 1e9 / dots
    );
}

/// The rows, in the order they run: (type, k, row bytes), each with the shape it stands for.
const ROWS: &[(GgmlType, usize, usize)] = &[
    // Row bytes: Q3_K 110 B/256, Q4_K 144 B/256, Q6_K 210 B/256.
    (GgmlType::Q3_K, 2048, (2048 / 256) * 110),
    (GgmlType::Q4_K, 2048, (2048 / 256) * 144),
    (GgmlType::Q6_K, 2048, (2048 / 256) * 210),
    // Qwen3.6/3.8 ffn_down_exps k = 512 (2 x 210 B = 420 B/row), the one-column twin of the
    // Q6_K tile rows at that shape.
    (GgmlType::Q6_K, 512, (512 / 256) * 210),
    // Real ffn_down_exps shape: k = 1408 (44 x 22 B = 968 B/row).
    (GgmlType::Q5_0, 1408, (1408 / 32) * 22),
    // Real ffn_down shape: k = 10944 (342 x 24 B = 8208 B/row).
    (GgmlType::Q5_1, 10944, (10944 / 32) * 24),
    // Q8_0 at the Q5_1 row's k, so the two compare per byte: 342 x 34 B = 11628 B/row.
    (GgmlType::Q8_0, 10944, (10944 / 32) * 34),
    // V4.1 ffn_down_exps shape (layers 0 and 1): k = 2304 (9 x 176 B = 1584 B/row).
    (GgmlType::Q5_K, 2304, (2304 / 256) * 176),
    // V4-Flash ffn_gate/up_exps shape: k = 4096 (16 x 98 B = 1568 B/row).
    (GgmlType::IQ3_XXS, 4096, (4096 / 256) * 98),
    // GLM-5.3-Flash UD-IQ4_XS ffn_gate/up_exps shape: k = 4096 (16 x 110 B = 1760 B/row),
    // beside the IQ3_XXS row at the same k.
    (GgmlType::IQ3_S, 4096, (4096 / 256) * 110),
    // GLM-5.3-Flash ffn_gate/up_exps row length k = 4096 at the IQ3_S row's k, so the three compare
    // per byte: Q2_K 16 x 84 B = 1344 B/row, IQ2_XS 16 x 74 B = 1184 B/row.
    (GgmlType::Q2_K, 4096, (4096 / 256) * 84),
    (GgmlType::IQ2_XS, 4096, (4096 / 256) * 74),
    // V4-Flash and MiMo-V2.6-Flash ffn_down_exps shape: k = 2048 (64 x 17 B = 1088 B/row).
    (GgmlType::MXFP4, 2048, (2048 / 32) * 17),
    // MiMo-V2.6-Flash ffn_gate/up_exps shape: k = 4096 (128 x 17 B = 2176 B/row).
    (GgmlType::MXFP4, 4096, (4096 / 32) * 17),
    // Qwen3.8 UD-Q3_K_XL ffn_down_exps shape (43 of 48 layers): k = 640 (20 x 18 B =
    // 360 B/row), beside the Q5_1 row it replaces at the same k.
    (GgmlType::IQ4_NL, 640, (640 / 32) * 18),
    (GgmlType::Q5_1, 640, (640 / 32) * 24),
    // Qwen3.8 UD-Q3_K_XL ffn_gate/up_exps shape (the one IQ4_XS layer): k = 2560
    // (10 x 136 B = 1360 B/row), beside the Q4_K row it replaces at the same k.
    (GgmlType::IQ4_XS, 2560, (2560 / 256) * 136),
    (GgmlType::Q4_K, 2560, (2560 / 256) * 144),
    // MiniMax-M3.1-Flash ffn_gate/up_exps shape (NVFP4): k = 6144 (384 x 9 B = 3456 B/row). The
    // host reads the planes form through the same kernel, one 16-value group at a time.
    (GgmlType::NVFP4_ROWS, 6144, (6144 / 16) * 9),
];

/// `qdot-rate [--only TYPE@K]...`: every row, or only the named ones (a counter run over one
/// row). A name that is no row is refused with the list.
fn main() {
    let rows: usize = 360_448;
    let name = |&(ty, k, _): &(GgmlType, usize, usize)| format!("{ty:?}@{k}");
    let mut only = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let ("--only", Some(v)) = (a.as_str(), args.next()) else {
            eprintln!("qdot-rate: usage: qdot-rate [--only TYPE@K]..., got {a:?}");
            std::process::exit(64);
        };
        if !ROWS.iter().any(|r| name(r) == v) {
            let all: Vec<String> = ROWS.iter().map(name).collect();
            eprintln!(
                "qdot-rate: --only {v}: no such row; the rows are {}",
                all.join(" ")
            );
            std::process::exit(64);
        }
        only.push(v);
    }
    for r in ROWS {
        if only.is_empty() || only.contains(&name(r)) {
            bench(r.0, r.1, r.2, rows);
        }
    }
    // The NVFP4 plane tile at M3.1's routed shape (k = 6144), C = 1 (the one-column kernel on this filler), 4 and 8.
    if only.is_empty() || only.iter().any(|v| v == "NVFP4_ROWS@6144") {
        for cols in [1, 4, qdot::TILE_COLS] {
            bench_planes_tile(6144, rows, cols);
        }
    }
    // The C = 8 tile at MXFP4's two shapes, the host union's run of eight columns: a full run only.
    if only.is_empty() {
        for k in [2048, 4096] {
            bench_tile(GgmlType::MXFP4, k, (k / 32) * 17, rows, qdot::TILE_COLS);
        }
        // The Q6_K tile at C = 4 and C = 8 at k = 2048 and k = 512, beside C = 1 on the same
        // filler: `dot_row_cols` with one column runs `dot_row`, the one-column kernel, so the
        // tile's ratio reads against C = 1, never against `ROWS`' random-`d` Q6_K lines.
        for k in [2048, 512] {
            for cols in [1, 4, qdot::TILE_COLS] {
                bench_tile(GgmlType::Q6_K, k, (k / 256) * 210, rows, cols);
            }
        }
    }
}
