//! qdot gates: every fused dot kernel against its scalar mirror, and against the
//! current path, the exact f64 dot or ik's dumped dot; the activation encoders
//! against ggml's round trip, their scalar twins and their edge cases.
//!
//! `hw_` tests need the box (AVX2, most of them a model file or harness dumps under
//! `$BLOOMERY_DATA/ref`) and are `#[ignore]`d; the pure tests (shape refusals,
//! encoder edge cases, the f32 helpers) run in the default set. `just gate-qdot`
//! runs both.
//!
//! The gates assert with `assert!` (not `debug_assert!`) because they run in release.

mod common;

use common::{q82_blocks, restore_q82x4};
use gguf::GgmlType;
use gguf::quant::{
    ActivationFormat, Nvfp4Rows, activation_format, dequant_row, half_to_f32,
    quantize_row_q8_0_roundtrip, quantize_row_q8_2_x4_roundtrip, quantize_row_q8_k_roundtrip,
};
#[cfg(target_arch = "x86_64")]
use qdot::dot_row_avx2;
use qdot::{
    QdotError, col_bytes, dot_row, dot_row_scalar, quantize_col, quantize_col_scalar, supports,
};
use std::fmt::Write as _;

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

/// The oracle's f32 dump of one tensor, byte length checked against expected value count.
fn oracle_f32(name: &str, expect: usize) -> Vec<f32> {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let path = format!("{base}/ref/{name}.0.f32");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(
        bytes.len(),
        expect * 4,
        "{path}: expected {expect} f32 values"
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Rows dotted per tensor.
const ROWS: usize = 1024;

struct Case<'a> {
    name: &'static str,
    k: usize,
    row_bytes: usize,
    /// First `ROWS` rows of the tensor.
    bytes: &'a [u8],
    /// Token 0's activation column.
    act: Vec<f32>,
    /// `act` in Q8_K blocks consumed by `dot_row`.
    acol: Vec<u8>,
}

fn load_case<'a>(
    g: &'a gguf::Gguf,
    name: &'static str,
    tensor: &str,
    k: usize,
    dump: &str,
) -> Case<'a> {
    let t = g
        .find(tensor)
        .unwrap_or_else(|| panic!("{tensor} not in the model"));
    assert_eq!(t.ty, GgmlType::Q3_K, "{tensor}: unexpected type");
    assert_eq!(
        t.dims[0] as usize, k,
        "{tensor}: dims[0] must be the row length"
    );
    let row_bytes = (k / 256) * 110;
    assert!(
        t.dims[1] as usize >= ROWS,
        "{tensor}: only {} rows, the gate needs {ROWS}",
        t.dims[1]
    );
    let data = g.data(t).unwrap();
    let bytes = &data[..ROWS * row_bytes];
    let vals = oracle_f32(dump, k * 6);
    let act = vals[..k].to_vec();
    let mut acol = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
    quantize_col(GgmlType::Q3_K, &act, &mut acol);
    Case {
        name,
        k,
        row_bytes,
        bytes,
        act,
        acol,
    }
}

/// Current path value: dequant_row + Q8_K roundtrip + sequential f32 dot.
fn current_path_dot(c: &Case, qa: &[f32], wbuf: &mut [f32], r: usize) -> f32 {
    dequant_row(GgmlType::Q3_K, &c.bytes[r * c.row_bytes..], wbuf).unwrap();
    let mut acc = 0.0f32;
    for i in 0..c.k {
        acc += wbuf[i] * qa[i];
    }
    acc
}

/// Exact f64 reference: integer codes decoded from raw bytes, summed in i64, scaled once in f64.
fn exact_dot_f64(wrow: &[u8], acol: &[u8], k: usize) -> f64 {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let nb = k / 256;
    let mut sum = 0.0f64;
    for sb in 0..nb {
        let blk = &wrow[sb * 110..sb * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let mut aux = [0u32; 3];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes(blk[96 + 4 * i..96 + 4 * i + 4].try_into().unwrap());
        }
        let word = |i: usize| match i {
            0 => (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4),
            1 => (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4),
            2 => ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4),
            _ => ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4),
        };
        // Unpacked scale byte 0..63, actual scale byte - 32.
        let scale = |idx: usize| (word(idx / 4).to_le_bytes()[idx % 4] as i8 - 32) as i64;
        let d = gguf::quant::half_to_f32(u16::from_le_bytes([blk[108], blk[109]])) as f64;
        let dcol = f32::from_le_bytes(acol[sb * 296..sb * 296 + 4].try_into().unwrap()) as f64;
        let q8 = &acol[sb * 296 + 8..sb * 296 + 8 + 256];
        let mut sumi = 0i64;
        let mut m = 1u8;
        for half in 0..2 {
            let mut shift = 0u32;
            for field in 0..4 {
                for h in 0..2 {
                    let sc = scale(8 * half + 2 * field + h);
                    for l in 0..16 {
                        let qv = ((qs[32 * half + 16 * h + l] >> shift) & 3) as i64;
                        let hv = if hmask[16 * h + l] & m != 0 { 0 } else { 4 };
                        sumi += sc
                            * (qv - hv)
                            * (q8[128 * half + 32 * field + 16 * h + l] as i8 as i64);
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
        sum += d * dcol * sumi as f64;
    }
    sum
}

// ------------------------------------------------------------- gate 1

/// Gate 1: restoring `d * q` from `quantize_col`'s packed codes and scale
/// must reproduce `quantize_row_q8_k_roundtrip`'s f32 output bit for bit.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_quantize_col_matches_roundtrip_bits() {
    for (dump, k) in [("ffn_norm-1", 2048usize), ("kv_compressed-1", 512)] {
        let vals = oracle_f32(dump, k * 6);
        let mut packed = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
        let mut rt = vec![0.0f32; k];
        let mut restored = vec![0.0f32; k];
        let mut bad = 0usize;
        // +0.0 vs -0.0: numerically equal with different bits; counted separately.
        let mut zero_sign = 0usize;
        let mut first: Option<(usize, usize, f32, f32)> = None;
        for t in 0..6 {
            let x = &vals[t * k..(t + 1) * k];
            quantize_col(GgmlType::Q3_K, x, &mut packed);
            quantize_row_q8_k_roundtrip(x, &mut rt);
            for b in 0..k / 256 {
                let blk = &packed[b * 296..b * 296 + 296];
                let d = f32::from_le_bytes(blk[0..4].try_into().unwrap());
                for j in 0..256 {
                    restored[b * 256 + j] = d * (blk[8 + j] as i8 as f32);
                }
            }
            for i in 0..k {
                if restored[i].to_bits() != rt[i].to_bits() {
                    if restored[i] == 0.0 && rt[i] == 0.0 {
                        zero_sign += 1;
                    } else {
                        bad += 1;
                        if first.is_none() {
                            first = Some((t, i, restored[i], rt[i]));
                        }
                    }
                }
            }
        }
        eprintln!(
            "gate1 {dump}: {k} values x 6 tokens = {} values, {bad} bit mismatches, {zero_sign} signed-zero pairs (+0 vs -0)",
            k * 6
        );
        if let Some((t, i, rv, tv)) = first {
            panic!(
                "{dump} token {t} index {i}: restored {rv:e} (bits {:#x}) != roundtrip {tv:e} (bits {:#x})",
                rv.to_bits(),
                tv.to_bits()
            );
        }
    }
}

// ------------------------------------------------------------- gate 2

/// Gate 2: `dot_row` matches the current path within relative difference <= 1e-5.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_dot_row_matches_current_path() {
    let g = gguf::Gguf::open(model_path()).unwrap();
    let cases = [
        load_case(
            &g,
            "ffn_gate_exps(k=2048)",
            "blk.1.ffn_gate_exps.weight",
            2048,
            "ffn_norm-1",
        ),
        load_case(
            &g,
            "attn_kv_b(k=512)",
            "blk.1.attn_kv_b.weight",
            512,
            "kv_compressed-1",
        ),
    ];
    for c in &cases {
        let mut qa = vec![0.0f32; c.k];
        quantize_row_q8_k_roundtrip(&c.act, &mut qa);
        let mut wbuf = vec![0.0f32; c.k];
        let mut max_abs = 0.0f64;
        let mut denom = 0.0f64;
        let mut max_row_rel = 0.0f64;
        let mut worst_row = 0usize;
        for r in 0..ROWS {
            let cur = current_path_dot(c, &qa, &mut wbuf, r);
            let fused = dot_row(GgmlType::Q3_K, &c.bytes[r * c.row_bytes..], &c.acol, c.k).unwrap();
            denom = denom.max(cur.abs() as f64);
            let d = (fused - cur) as f64;
            max_abs = max_abs.max(d.abs());
            let rel = if cur != 0.0 {
                d / cur as f64
            } else {
                f64::INFINITY
            };
            if rel > max_row_rel {
                max_row_rel = rel;
                worst_row = r;
            }
        }
        let global_rel = max_abs / denom;
        eprintln!(
            "gate2 {:20} rows={ROWS} max|diff|={max_abs:.4e} max|current|={denom:.4e} global_rel={global_rel:.3e} (gate 1e-5), worst row_rel={max_row_rel:.3e} @ row {worst_row}",
            c.name
        );
        assert!(
            global_rel <= 1e-5,
            "{}: fused vs current path global rel {global_rel:.3e} > 1e-5",
            c.name
        );
    }
}

// ------------------------------------------------------------- gate 3

/// Gate 3: fused kernel lands closer to exact f64 than current f32 path on >50% of rows.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_dot_row_closer_to_exact_than_current_path() {
    let g = gguf::Gguf::open(model_path()).unwrap();
    let cases = [
        load_case(
            &g,
            "ffn_gate_exps(k=2048)",
            "blk.1.ffn_gate_exps.weight",
            2048,
            "ffn_norm-1",
        ),
        load_case(
            &g,
            "attn_kv_b(k=512)",
            "blk.1.attn_kv_b.weight",
            512,
            "kv_compressed-1",
        ),
    ];
    for c in &cases {
        let mut qa = vec![0.0f32; c.k];
        quantize_row_q8_k_roundtrip(&c.act, &mut qa);
        let mut wbuf = vec![0.0f32; c.k];
        let mut closer = 0usize;
        let mut ties = 0usize;
        let mut worse = 0usize;
        let mut worst_fused_rel = 0.0f64;
        let mut worst_cur_rel = 0.0f64;
        for r in 0..ROWS {
            let exact = exact_dot_f64(&c.bytes[r * c.row_bytes..], &c.acol, c.k);
            let fused =
                dot_row(GgmlType::Q3_K, &c.bytes[r * c.row_bytes..], &c.acol, c.k).unwrap() as f64;
            let cur = current_path_dot(c, &qa, &mut wbuf, r) as f64;
            let fe = (fused - exact).abs();
            let ce = (cur - exact).abs();
            if fe < ce {
                closer += 1;
            } else if fe == ce {
                ties += 1;
            } else {
                worse += 1;
            }
            worst_fused_rel = worst_fused_rel.max(fe / exact.abs());
            worst_cur_rel = worst_cur_rel.max(ce / exact.abs());
        }
        eprintln!(
            "gate3 {:20} closer={closer}/{ROWS} (ties {ties}, worse {worse}) worst_rel fused={worst_fused_rel:.3e} current={worst_cur_rel:.3e}",
            c.name
        );
        assert!(
            closer > ROWS / 2,
            "{}: fused closer on only {closer}/{ROWS} rows (ties {ties}, worse {worse}) — the fused kernel is expected to beat the f32 path on most rows; if it does not, the kernel is wrong",
            c.name
        );
        assert!(
            worst_fused_rel <= worst_cur_rel,
            "{}: fused worst rel {worst_fused_rel:.3e} exceeds current path worst rel {worst_cur_rel:.3e}",
            c.name
        );
    }
}

// ------------------------------------------------------------- gate 4

/// Gate 4: scalar fallback is bit-identical to the AVX2 kernel.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_scalar_fallback_bit_identical_to_avx2() {
    assert!(
        supports(GgmlType::Q3_K),
        "gate 4 compares the AVX2 kernel against the scalar mirror; this CPU has no AVX2"
    );
    let g = gguf::Gguf::open(model_path()).unwrap();
    let cases = [
        load_case(
            &g,
            "ffn_gate_exps(k=2048)",
            "blk.1.ffn_gate_exps.weight",
            2048,
            "ffn_norm-1",
        ),
        load_case(
            &g,
            "attn_kv_b(k=512)",
            "blk.1.attn_kv_b.weight",
            512,
            "kv_compressed-1",
        ),
    ];
    for c in &cases {
        for r in 0..ROWS {
            let row = &c.bytes[r * c.row_bytes..];
            let a = dot_row_avx2(GgmlType::Q3_K, row, &c.acol, c.k).unwrap();
            let b = dot_row_scalar(GgmlType::Q3_K, row, &c.acol, c.k).unwrap();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{} row {r}: AVX2 {a:e} (bits {:#x}) != scalar {b:e} (bits {:#x})",
                c.name,
                a.to_bits(),
                b.to_bits()
            );
        }
        eprintln!(
            "gate4 {:20} {ROWS} rows bit-identical (avx2 == scalar)",
            c.name
        );
    }
}

// ------------------------------------------------------------- gate 5

/// Gate 5 (pure): unaligned k is rejected with Err, and buffer size contracts hold.
#[test]
fn rejects_unaligned_k() {
    // Buffers sized for k = 2048.
    let wrow = vec![0u8; 110 * 8];
    let acol = vec![0u8; 296 * 8];
    for k in [1usize, 100, 255, 2048 + 128] {
        match dot_row(GgmlType::Q3_K, &wrow, &acol, k) {
            Err(QdotError::UnalignedK { k: got, .. }) => assert_eq!(got, k),
            other => panic!("k = {k}: expected UnalignedK, got {other:?}"),
        }
    }
    // Unsupported type refuses before reading bytes.
    assert!(matches!(
        dot_row(GgmlType::F16, &wrow, &acol, 2048),
        Err(QdotError::UnsupportedType(GgmlType::F16))
    ));
    // Short buffers are errors, not reads past the slice.
    assert!(matches!(
        dot_row(GgmlType::Q3_K, &wrow[..879], &acol, 2048),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q3_K, &wrow, &acol[..295], 2048),
        Err(QdotError::ShortActivationCol { .. })
    ));
    // Aligned k on valid buffers succeeds.
    assert_eq!(dot_row(GgmlType::Q3_K, &wrow, &acol, 2048).unwrap(), 0.0);
    // col_bytes allocation contracts per type.
    assert_eq!(col_bytes(GgmlType::Q3_K, 2048), 296 * 8);
    assert_eq!(col_bytes(GgmlType::Q3_K, 512), 296 * 2);
    // Q4_K pairs q8_2_x4: 144 bytes per 128 values.
    assert_eq!(col_bytes(GgmlType::Q4_K, 2048), 144 * 16);
    assert_eq!(col_bytes(GgmlType::Q4_K, 512), 144 * 4);
    let wrow4 = vec![0u8; 144 * 8];
    let acol4 = vec![0u8; 144 * 16];
    assert!(matches!(
        dot_row(GgmlType::Q4_K, &wrow4, &acol4[..2303], 2048),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(dot_row(GgmlType::Q4_K, &wrow4, &acol4, 2048).unwrap(), 0.0);
    // Q6_K pairs q8_2_x4 with 210 weight bytes per 256 values.
    assert_eq!(col_bytes(GgmlType::Q6_K, 2048), 144 * 16);
    assert_eq!(col_bytes(GgmlType::Q6_K, 512), 144 * 4);
    let wrow6 = vec![0u8; 210 * 8];
    let acol6 = vec![0u8; 144 * 16];
    assert!(matches!(
        dot_row(GgmlType::Q6_K, &wrow6[..1679], &acol6, 2048),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q6_K, &wrow6, &acol6[..2303], 2048),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(dot_row(GgmlType::Q6_K, &wrow6, &acol6, 2048).unwrap(), 0.0);
    // Q5_0 uses 32-value blocks (22 bytes per 32 values).
    assert_eq!(col_bytes(GgmlType::Q5_0, 1408), 144 * 11);
    assert_eq!(col_bytes(GgmlType::Q5_0, 160), 144 + 36);
    assert_eq!(col_bytes(GgmlType::Q5_0, 192), 144 + 2 * 36);
    let wrow5 = vec![0u8; 22 * 44];
    let acol5 = vec![0u8; 144 * 11];
    assert_eq!(dot_row(GgmlType::Q5_0, &wrow5, &acol5, 1408).unwrap(), 0.0);
    assert!(matches!(
        dot_row(GgmlType::Q5_0, &wrow5, &acol5, 100),
        Err(QdotError::UnalignedK { k: 100, gran: 32 })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q5_0, &wrow5[..21], &acol5, 1408),
        Err(QdotError::ShortWeightRow { .. })
    ));
    // Q5_1 uses 32-value blocks (24 bytes per 32 values).
    assert_eq!(col_bytes(GgmlType::Q5_1, 10944), 144 * 85 + 2 * 36);
    assert_eq!(col_bytes(GgmlType::Q5_1, 160), 144 + 36);
    let wrow51 = vec![0u8; 24 * 342];
    let acol51 = vec![0u8; 144 * 85 + 2 * 36];
    assert_eq!(
        dot_row(GgmlType::Q5_1, &wrow51, &acol51, 10944).unwrap(),
        0.0
    );
    assert!(matches!(
        dot_row(GgmlType::Q5_1, &wrow51, &acol51, 100),
        Err(QdotError::UnalignedK { k: 100, gran: 32 })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q5_1, &wrow51[..23], &acol51, 10944),
        Err(QdotError::ShortWeightRow { .. })
    ));
    // Q5_K pairs q8_2_x4 with 176 weight bytes per 256 values (k = 2304, the V4.1 down rows).
    assert_eq!(col_bytes(GgmlType::Q5_K, 2304), 144 * 18);
    let wrow5k = vec![0u8; 176 * 9];
    let acol5k = vec![0u8; 144 * 18];
    assert!(matches!(
        dot_row(GgmlType::Q5_K, &wrow5k, &acol5k, 2304 + 32),
        Err(QdotError::UnalignedK { k: 2336, gran: 256 })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q5_K, &wrow5k[..176 * 9 - 1], &acol5k, 2304),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q5_K, &wrow5k, &acol5k[..144 * 18 - 1], 2304),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::Q5_K, &wrow5k, &acol5k, 2304).unwrap(),
        0.0
    );
    // IQ3_XXS pairs q8_K with 98 weight bytes per 256 values (k = 4096, the V4 gate/up rows).
    assert_eq!(col_bytes(GgmlType::IQ3_XXS, 4096), 296 * 16);
    let wrow3x = vec![0u8; 98 * 16];
    let acol3x = vec![0u8; 296 * 16];
    assert!(matches!(
        dot_row(GgmlType::IQ3_XXS, &wrow3x, &acol3x, 4096 + 32),
        Err(QdotError::UnalignedK { k: 4128, gran: 256 })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ3_XXS, &wrow3x[..98 * 16 - 1], &acol3x, 4096),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ3_XXS, &wrow3x, &acol3x[..296 * 16 - 1], 4096),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::IQ3_XXS, &wrow3x, &acol3x, 4096).unwrap(),
        0.0
    );
    // MXFP4 uses 32-value blocks (17 bytes per 32 values) and q8_2 tails past the x4 groups.
    assert_eq!(col_bytes(GgmlType::MXFP4, 2048), 144 * 16);
    assert_eq!(col_bytes(GgmlType::MXFP4, 160), 144 + 36);
    let wrowmx = vec![0u8; 17 * 64];
    let acolmx = vec![0u8; 144 * 16];
    assert!(matches!(
        dot_row(GgmlType::MXFP4, &wrowmx, &acolmx, 100),
        Err(QdotError::UnalignedK { k: 100, gran: 32 })
    ));
    assert!(matches!(
        dot_row(GgmlType::MXFP4, &wrowmx[..17 * 64 - 1], &acolmx, 2048),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::MXFP4, &wrowmx, &acolmx, 2048).unwrap(),
        0.0
    );
    // Q8_0 uses 32-value blocks (34 bytes per 32 values) and q8_2 tails past the x4 groups:
    // k = 2144 is 16 groups and three tail blocks.
    assert_eq!(col_bytes(GgmlType::Q8_0, 2144), 144 * 16 + 3 * 36);
    assert_eq!(col_bytes(GgmlType::Q8_0, 480), 144 * 3 + 3 * 36);
    let wrow80 = vec![0u8; 34 * 67];
    let acol80 = vec![0u8; 144 * 16 + 3 * 36];
    assert!(matches!(
        dot_row(GgmlType::Q8_0, &wrow80, &acol80, 100),
        Err(QdotError::UnalignedK { k: 100, gran: 32 })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q8_0, &wrow80[..34 * 67 - 1], &acol80, 2144),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(
            GgmlType::Q8_0,
            &wrow80,
            &acol80[..144 * 16 + 3 * 36 - 1],
            2144
        ),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::Q8_0, &wrow80, &acol80, 2144).unwrap(),
        0.0
    );
    // IQ4_NL uses 32-value blocks (18 bytes per 32 values) and q8_2 tails past the x4
    // groups: k = 640 is the Qwen3.8 down row (20 blocks, 5 whole groups, no tail).
    assert_eq!(col_bytes(GgmlType::IQ4_NL, 640), 144 * 5);
    assert_eq!(col_bytes(GgmlType::IQ4_NL, 672), 144 * 5 + 36);
    let wrownl = vec![0u8; 18 * 21];
    let acolnl = vec![0u8; 144 * 5 + 36];
    assert!(matches!(
        dot_row(GgmlType::IQ4_NL, &wrownl, &acolnl, 100),
        Err(QdotError::UnalignedK { k: 100, gran: 32 })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ4_NL, &wrownl[..18 * 21 - 1], &acolnl, 672),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ4_NL, &wrownl, &acolnl[..144 * 5 + 35], 672),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::IQ4_NL, &wrownl, &acolnl, 672).unwrap(),
        0.0
    );
    // IQ4_XS pairs q8_K with 136 weight bytes per 256 values (k = 2560, the Qwen3.8
    // gate/up rows of the one IQ4_XS layer).
    assert_eq!(col_bytes(GgmlType::IQ4_XS, 2560), 296 * 10);
    let wrowxs = vec![0u8; 136 * 10];
    let acolxs = vec![0u8; 296 * 10];
    assert!(matches!(
        dot_row(GgmlType::IQ4_XS, &wrowxs, &acolxs, 2560 + 32),
        Err(QdotError::UnalignedK { k: 2592, gran: 256 })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ4_XS, &wrowxs[..136 * 10 - 1], &acolxs, 2560),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ4_XS, &wrowxs, &acolxs[..296 * 10 - 1], 2560),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::IQ4_XS, &wrowxs, &acolxs, 2560).unwrap(),
        0.0
    );
    // IQ3_S pairs q8_K with 110 weight bytes per 256 values (k = 4096, GLM-5.3-Flash
    // UD-IQ4_XS's gate/up rows).
    assert_eq!(col_bytes(GgmlType::IQ3_S, 4096), 296 * 16);
    let wrow3s = vec![0u8; 110 * 16];
    let acol3s = vec![0u8; 296 * 16];
    assert!(matches!(
        dot_row(GgmlType::IQ3_S, &wrow3s, &acol3s, 4096 + 32),
        Err(QdotError::UnalignedK { k: 4128, gran: 256 })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ3_S, &wrow3s[..110 * 16 - 1], &acol3s, 4096),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ3_S, &wrow3s, &acol3s[..296 * 16 - 1], 4096),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::IQ3_S, &wrow3s, &acol3s, 4096).unwrap(),
        0.0
    );
    // Q2_K pairs q8_K with 84 weight bytes per 256 values.
    assert_eq!(col_bytes(GgmlType::Q2_K, 4096), 296 * 16);
    let wrow2k = vec![0u8; 84 * 16];
    let acol2k = vec![0u8; 296 * 16];
    assert!(matches!(
        dot_row(GgmlType::Q2_K, &wrow2k, &acol2k, 4096 + 32),
        Err(QdotError::UnalignedK { k: 4128, gran: 256 })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q2_K, &wrow2k[..84 * 16 - 1], &acol2k, 4096),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q2_K, &wrow2k, &acol2k[..296 * 16 - 1], 4096),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::Q2_K, &wrow2k, &acol2k, 4096).unwrap(),
        0.0
    );
    // IQ2_XS pairs q8_K with 74 weight bytes per 256 values.
    assert_eq!(col_bytes(GgmlType::IQ2_XS, 4096), 296 * 16);
    let wrow2xs = vec![0u8; 74 * 16];
    let acol2xs = vec![0u8; 296 * 16];
    assert!(matches!(
        dot_row(GgmlType::IQ2_XS, &wrow2xs, &acol2xs, 4096 + 32),
        Err(QdotError::UnalignedK { k: 4128, gran: 256 })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ2_XS, &wrow2xs[..74 * 16 - 1], &acol2xs, 4096),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::IQ2_XS, &wrow2xs, &acol2xs[..296 * 16 - 1], 4096),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::IQ2_XS, &wrow2xs, &acol2xs, 4096).unwrap(),
        0.0
    );
    // Q2_0 pairs q8_0 (34 bytes per 32 values) with 18 weight bytes per 64 values.
    assert_eq!(col_bytes(GgmlType::Q2_0, 2048), 34 * 64);
    let wrow20 = vec![0u8; 18 * 32];
    let acol20 = vec![0u8; 34 * 64];
    assert!(matches!(
        dot_row(GgmlType::Q2_0, &wrow20, &acol20, 2048 + 32),
        Err(QdotError::UnalignedK { k: 2080, gran: 64 })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q2_0, &wrow20[..18 * 32 - 1], &acol20, 2048),
        Err(QdotError::ShortWeightRow { .. })
    ));
    assert!(matches!(
        dot_row(GgmlType::Q2_0, &wrow20, &acol20[..34 * 64 - 1], 2048),
        Err(QdotError::ShortActivationCol { .. })
    ));
    assert_eq!(
        dot_row(GgmlType::Q2_0, &wrow20, &acol20, 2048).unwrap(),
        0.0
    );
    // Supported type table check.
    assert!(!supports(GgmlType::F16));
    assert!(supports(GgmlType::Q5_K));
    assert!(supports(GgmlType::Q5_1));
    assert!(supports(GgmlType::IQ3_XXS));
    assert!(supports(GgmlType::IQ3_S));
    assert!(supports(GgmlType::Q2_K));
    assert!(supports(GgmlType::IQ2_XS));
    assert!(supports(GgmlType::MXFP4));
    assert!(supports(GgmlType::Q8_0));
    assert!(supports(GgmlType::IQ4_NL));
    assert!(supports(GgmlType::IQ4_XS));
    assert!(supports(GgmlType::Q2_0));
}

/// `col_bytes` panics on unaligned k.
#[test]
#[should_panic(expected = "whole 256-value super-blocks")]
fn col_bytes_rejects_unaligned_k() {
    let _ = col_bytes(GgmlType::Q3_K, 100);
}

/// Parse ik kernel reference dump: tensor header, hex activation bytes, and output rows.
fn parse_ik_dot_dump(dump: &str) -> (usize, Vec<u8>, Vec<u32>) {
    let mut dump_k = 0usize;
    let mut rows = Vec::new();
    let mut acol: Vec<u8> = Vec::new();
    for line in dump.lines() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next(), it.next(), it.next()) {
            (Some("tensor"), Some(_), Some("k"), Some(kk)) => dump_k = kk.parse().unwrap(),
            (Some("row"), Some(_), Some(hex), None) => {
                rows.push(u32::from_str_radix(hex, 16).expect("ik dumps raw f32 bits"));
            }
            // Hex-encoded activation bytes.
            (Some(hex), None, None, None) if hex.len() > 64 => {
                let b = hex.as_bytes();
                acol.extend(
                    b.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap()),
                );
            }
            _ => {}
        }
    }
    (dump_k, acol, rows)
}

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_q4k_dot_row_matches_scalar_and_predicts() {
    // Gate triple for Q4_K x Q8_2_X4: encoder bit-identical, kernel vs emulator bit-identical, within 1 ULP of reference.
    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| w.ty == gguf::GgmlType::Q4_K && (w.dims[0] as usize).is_multiple_of(256))
        .expect("the model must carry at least one aligned Q4_K tensor");
    let k = w.dims[0] as usize;
    let n = w.dims[1] as usize;

    let bytes = g.data(w).unwrap();
    let row_bytes = bytes.len() / n;
    assert_eq!(row_bytes, 144 * k / 256);

    // Real activations: the oracle's first-block normed input, first column.
    let vals = oracle_f32("attn_norm-0", 2048 * 6);
    let xs: Vec<f32> = vals[..k].to_vec();
    let cb = qdot::col_bytes(gguf::GgmlType::Q4_K, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(gguf::GgmlType::Q4_K, &xs, &mut acol);

    // Reference dump from ik's kernel on this tensor.
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q4k-x4-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    assert_eq!(
        ik_acol.len(),
        cb,
        "ik's q8_2_x4 column must size-match ours"
    );

    // Gate 0: encoder bit-identical to reference.
    for (i, (a, b)) in ik_acol.iter().zip(&acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!("gate 0: {cb} encoder bytes bit-identical to ik's quantize_row_q8_2_x4");

    // Gate A: bit identity between kernel and scalar emulator.
    let rows = n.min(1024);
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let a = qdot::dot_row(gguf::GgmlType::Q4_K, src, &acol, k).unwrap();
        let b = qdot::dot_row_scalar(gguf::GgmlType::Q4_K, src, &acol, k).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: kernel and emulator must be bit-identical"
        );
    }
    eprintln!("gate A: {rows} rows bit-identical (kernel vs emulator)");

    // Gate B: matches reference kernel within 1 ULP.
    assert!(want.len() >= 64, "dump needs 64 rows");
    for (r, &w) in want.iter().take(64).enumerate() {
        let got = qdot::dot_row(
            gguf::GgmlType::Q4_K,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &ik_acol,
            k,
        )
        .unwrap();
        let ulp = (got.to_bits() as i64 - w as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(w)
        );
    }
    eprintln!(
        "gate B: 64 rows within 1 ULP of ik's mul_mat_qX_K_q8_2_X4_T (on ik's own activations)"
    );
}

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_q6k_dot_row_matches_scalar_and_predicts() {
    // Gate triple for Q6_K x Q8_2_X4: encoder bit-identical, kernel vs emulator bit-identical, within 1 ULP of reference.
    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| w.ty == gguf::GgmlType::Q6_K && (w.dims[0] as usize).is_multiple_of(256))
        .expect("the model must carry at least one aligned Q6_K tensor");
    let k = w.dims[0] as usize;
    let n = w.dims[1] as usize;

    let bytes = g.data(w).unwrap();
    let row_bytes = bytes.len() / n;
    assert_eq!(row_bytes, 210 * k / 256);

    // Real activations: the oracle's first-block normed input, first column.
    let vals = oracle_f32("attn_norm-0", 2048 * 6);
    let xs: Vec<f32> = vals[..k].to_vec();
    let cb = qdot::col_bytes(gguf::GgmlType::Q6_K, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(gguf::GgmlType::Q6_K, &xs, &mut acol);

    // Reference dump from ik's kernel on this tensor.
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q6k-x4-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    assert_eq!(
        ik_acol.len(),
        cb,
        "ik's q8_2_x4 column must size-match ours"
    );

    // Gate 0: encoder bit-identical to reference.
    for (i, (a, b)) in ik_acol.iter().zip(&acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!("gate 0: {cb} encoder bytes bit-identical to ik's quantize_row_q8_2_x4");

    // Gate A: bit identity between kernel and scalar emulator.
    let rows = n.min(1024);
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let a = qdot::dot_row(gguf::GgmlType::Q6_K, src, &acol, k).unwrap();
        let b = qdot::dot_row_scalar(gguf::GgmlType::Q6_K, src, &acol, k).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: kernel and emulator must be bit-identical"
        );
    }
    eprintln!("gate A: {rows} rows bit-identical (kernel vs emulator)");

    // Gate B: matches reference kernel within 1 ULP.
    assert!(want.len() >= 64, "dump needs 64 rows");
    for (r, &w) in want.iter().take(64).enumerate() {
        let got = qdot::dot_row(
            gguf::GgmlType::Q6_K,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &ik_acol,
            k,
        )
        .unwrap();
        let ulp = (got.to_bits() as i64 - w as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(w)
        );
    }
    eprintln!(
        "gate B: 64 rows within 1 ULP of ik's mul_mat_qY_K_q8_2_X4_T (on ik's own activations)"
    );
}

/// Gate triple for Q5_0 x Q8_2_X4: encoder bit-identical, kernel vs emulator bit-identical, within 1 ULP of reference.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_q5f0_dot_row_matches_scalar_and_predicts_ik() {
    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| {
            w.ty == gguf::GgmlType::Q5_0
                && (w.dims[0] as usize).is_multiple_of(32)
                && (w.dims[0] as usize).is_multiple_of(128)
        })
        .expect("the model must carry at least one aligned Q5_0 tensor");
    let k = w.dims[0] as usize;
    // ffn_down_exps is a 3-D stack (k, rows_per_expert, 64 experts).
    let n: usize = w.dims[1..].iter().product::<u64>() as usize;

    let bytes = g.data(w).unwrap();
    let row_bytes = bytes.len() / n;
    assert_eq!(row_bytes, 22 * k / 32);

    // Real activations: the oracle's first-block normed input, first column.
    // attn_norm-0 is 2048 values per token; k = 1408 fits inside it.
    let vals = oracle_f32("attn_norm-0", 2048 * 6);
    let xs: Vec<f32> = vals[..k].to_vec();
    let cb = qdot::col_bytes(gguf::GgmlType::Q5_0, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(gguf::GgmlType::Q5_0, &xs, &mut acol);

    // Reference dump from ik's kernel on this tensor.
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q5f0-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    assert_eq!(
        ik_acol.len(),
        cb,
        "ik's q8_2_x4 column must size-match ours"
    );

    // Gate 0: encoder bit-identical to reference.
    for (i, (a, b)) in ik_acol.iter().zip(&acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!("gate 0: {cb} encoder bytes bit-identical to ik's quantize_row_q8_2_x4");

    // Gate A: bit identity between kernel and scalar emulator.
    let rows = n.min(1024);
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let a = qdot::dot_row(gguf::GgmlType::Q5_0, src, &acol, k).unwrap();
        let b = qdot::dot_row_scalar(gguf::GgmlType::Q5_0, src, &acol, k).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: kernel and emulator must be bit-identical"
        );
    }
    eprintln!("gate A: {rows} rows bit-identical (kernel vs emulator)");

    // Gate B: matches reference kernel within 1 ULP.
    assert!(want.len() >= 64, "dump needs 64 rows");
    for (r, &w) in want.iter().take(64).enumerate() {
        let got = qdot::dot_row(
            gguf::GgmlType::Q5_0,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &ik_acol,
            k,
        )
        .unwrap();
        let ulp = (got.to_bits() as i64 - w as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(w)
        );
    }
    eprintln!("gate B: 64 rows within 1 ULP of ik's mul_mat_qX_1_q8_2_T (on ik's own activations)");
}

/// Gate triple for Q5_1 x Q8_2_X4: encoder bit-identical, kernel vs emulator bit-identical, within 1 ULP of reference.
#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_q5f1_dot_row_matches_scalar_and_predicts_ik() {
    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| w.ty == gguf::GgmlType::Q5_1 && (w.dims[0] as usize).is_multiple_of(32))
        .expect("the model must carry the Q5_1 tensor (blk.0.ffn_down)");
    let k = w.dims[0] as usize;
    let n = w.dims[1] as usize;
    assert_eq!(k, 10944, "the model's Q5_1 site is the dense ffn_down");

    let bytes = g.data(w).unwrap();
    let row_bytes = bytes.len() / n;
    assert_eq!(row_bytes, 24 * k / 32);

    // Real activations from oracle ffn_up_gate dump.
    let vals = oracle_f32("ffn_up_gate-0", k * 6);
    let xs: Vec<f32> = vals[..k].to_vec();
    let cb = qdot::col_bytes(gguf::GgmlType::Q5_1, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(gguf::GgmlType::Q5_1, &xs, &mut acol);

    // Reference dump from ik's kernel on this tensor.
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q5f1-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    assert_eq!(
        ik_acol.len(),
        cb,
        "ik's q8_2_x4 column must size-match ours"
    );

    // Gate 0: encoder bit-identical to reference on tail blocks.
    for (i, (a, b)) in ik_acol.iter().zip(&acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!(
        "gate 0: {cb} encoder bytes bit-identical to ik's quantize_row_q8_2_x4 (incl. 2 tail blocks)"
    );

    // Gate A: bit identity between kernel and scalar emulator.
    let rows = n.min(1024);
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let a = qdot::dot_row(gguf::GgmlType::Q5_1, src, &acol, k).unwrap();
        let b = qdot::dot_row_scalar(gguf::GgmlType::Q5_1, src, &acol, k).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: kernel and emulator must be bit-identical"
        );
    }
    eprintln!("gate A: {rows} rows bit-identical (kernel vs emulator)");

    // Gate B: matches reference kernel within 1 ULP.
    assert!(want.len() >= 64, "dump needs 64 rows");
    for (r, &w) in want.iter().take(64).enumerate() {
        let got = qdot::dot_row(
            gguf::GgmlType::Q5_1,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &ik_acol,
            k,
        )
        .unwrap();
        let ulp = (got.to_bits() as i64 - w as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(w)
        );
    }
    eprintln!("gate B: 64 rows within 1 ULP of ik's mul_mat_qX_1_q8_2_T (on ik's own activations)");
}

// ------------------------------------------------------- Q5_K x Q8_2_X4
// The reference model carries no Q5_K tensor; the V4.1 file's first shard holds
// blk.0.ffn_down_exps.weight (k = 2304). The strict `Gguf::open` refuses that shard (its
// token_embd is bf16), so the tensor is found through the header-only inventory and its rows
// are read at `data_base + offset`. Gates 0, A and B are the triple the other x4 types carry,
// one test each so that one broken kernel shows every gate it breaks.

/// The first Q5_K tensor with k % 256 == 0 — the scan `tools/ref/q5k_x4_ref.cpp` does.
struct Q5kCase {
    name: String,
    k: usize,
    row_bytes: usize,
    /// The tensor's first rows.
    bytes: Vec<u8>,
    /// The column the ik harness codes, through `quantize_col`.
    acol: Vec<u8>,
}

impl Q5kCase {
    fn row(&self, r: usize) -> &[u8] {
        &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes]
    }
}

fn load_q5k(rows: usize) -> Q5kCase {
    use std::os::unix::fs::FileExt;
    // The V4.1 first shard ([`gguf::v41::model`]); `BLOOMERY_Q5K_MODEL` overrides it.
    let path = std::env::var("BLOOMERY_Q5K_MODEL").unwrap_or_else(|_| gguf::v41::model());
    let inv = gguf::inventory_of(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let t = inv
        .tensors
        .iter()
        .find(|t| t.type_id == GgmlType::Q5_K.as_u32() && t.dims[0].is_multiple_of(256))
        .unwrap_or_else(|| panic!("{path} carries no aligned Q5_K tensor"));
    let k = t.dims[0] as usize;
    let n = t.dims[1..].iter().product::<u64>() as usize;
    let row_bytes = 176 * k / 256;
    assert_eq!(t.nbytes, Some((n * row_bytes) as u64), "{}: bytes", t.name);
    assert!(
        n >= rows,
        "{}: only {n} rows, the gate needs {rows}",
        t.name
    );
    let mut bytes = vec![0u8; rows * row_bytes];
    std::fs::File::open(&path)
        .and_then(|f| f.read_exact_at(&mut bytes, inv.data_base + t.offset))
        .unwrap_or_else(|e| panic!("{path}: read {}: {e}", t.name));
    // The attn_norm-0 dump is 2048 values per token: the first k = 2304 are token 0 and the
    // start of token 1, the column the ik harness reads.
    let xs = oracle_f32("attn_norm-0", 2048 * 6)[..k].to_vec();
    let mut acol = vec![0u8; col_bytes(GgmlType::Q5_K, k)];
    quantize_col(GgmlType::Q5_K, &xs, &mut acol);
    Q5kCase {
        name: t.name.clone(),
        k,
        row_bytes,
        bytes,
        acol,
    }
}

/// ik's dump for the Q5_K tensor: its activation bytes and its 64 row results.
fn q5k_ik_dump(c: &Q5kCase) -> (Vec<u8>, Vec<u32>) {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q5k-x4-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, c.k,
        "the dump and this scan must land on the same tensor"
    );
    assert_eq!(
        ik_acol.len(),
        c.acol.len(),
        "ik's q8_2_x4 column must size-match ours"
    );
    (ik_acol, want)
}

/// Q5_K gate 0: `quantize_col` codes the column byte for byte as ik's `quantize_row_q8_2_x4`.
#[test]
#[ignore = "hw: needs the box, the V4.1 shard and $BLOOMERY_DATA/ref"]
fn hw_q5k_encoder_matches_ik() {
    let c = load_q5k(1);
    let (ik_acol, _) = q5k_ik_dump(&c);
    for (i, (a, b)) in ik_acol.iter().zip(&c.acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!(
        "q5k gate 0: {} encoder bytes bit-identical to ik's quantize_row_q8_2_x4 ({}, k = {})",
        c.acol.len(),
        c.name,
        c.k
    );
}

/// Q5_K gate A: the AVX2 kernel and its emulator agree bit for bit on real rows.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box, the V4.1 shard and $BLOOMERY_DATA/ref"]
fn hw_q5k_kernel_matches_emulator() {
    assert!(
        supports(GgmlType::Q5_K),
        "gate A compares the AVX2 kernel against its emulator; this CPU has no AVX2+FMA"
    );
    let c = load_q5k(ROWS);
    for r in 0..ROWS {
        let a = dot_row_avx2(GgmlType::Q5_K, c.row(r), &c.acol, c.k).unwrap();
        let b = dot_row_scalar(GgmlType::Q5_K, c.row(r), &c.acol, c.k).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: kernel {a:e} (bits {:#x}) and emulator {b:e} (bits {:#x}) must be bit-identical",
            a.to_bits(),
            b.to_bits()
        );
    }
    eprintln!(
        "q5k gate A: {ROWS} rows bit-identical (kernel vs emulator, {})",
        c.name
    );
}

/// Q5_K gate B: on ik's own activations the kernel is within 1 ULP of ik's
/// `mul_mat_qX_K_q8_2_X4_T<DequantizerQ5K_AVX2>` on all 64 dumped rows.
#[test]
#[ignore = "hw: needs the box, the V4.1 shard and $BLOOMERY_DATA/ref"]
fn hw_q5k_kernel_predicts_ik() {
    let c = load_q5k(64);
    let (ik_acol, want) = q5k_ik_dump(&c);
    assert!(want.len() >= 64, "dump needs 64 rows");
    let mut worst = 0i64;
    let mut exact = 0usize;
    for (r, &w) in want.iter().take(64).enumerate() {
        let got = dot_row(GgmlType::Q5_K, c.row(r), &ik_acol, c.k).unwrap();
        let ulp = (got.to_bits() as i64 - w as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(w)
        );
        worst = worst.max(ulp);
        exact += usize::from(ulp == 0);
    }
    eprintln!(
        "q5k gate B: 64 rows within 1 ULP of ik's mul_mat_qX_K_q8_2_X4_T<DequantizerQ5K_AVX2> \
         (on ik's own activations): max {worst} ULP, {exact}/64 rows at 0 ULP"
    );
}

/// Q5_K dequantization: `dequant_row` equals ggml's own `to_float` (dumped by q5k_x4_ref.cpp)
/// on the tensor's first rows, bit for bit — the transcription keeps ggml's operation order,
/// including the multiply-subtract the compiled library fuses. Then the kernel against an f64
/// dot of those rows with the restored column, in gate 2's band: max |diff| / max |ref| <= 1e-5.
#[test]
#[ignore = "hw: needs the box, the V4.1 shard and $BLOOMERY_DATA/ref"]
fn hw_q5k_dequant_matches_ggml() {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let meta = std::fs::read_to_string(format!("{base}/ref/q5k-v41-dequant.meta"))
        .expect("run just build-ref first (build-qdot-ref.sh writes the q5_K dequant dump)");
    let field = |key: &str| {
        meta.lines()
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or_else(|| panic!("q5k-v41-dequant.meta has no {key}"))
    };
    let rows: usize = field("rows=").parse().unwrap();
    let c = load_q5k(rows);
    assert_eq!(
        field("tensor="),
        c.name,
        "the dump and this scan must land on the same tensor"
    );
    assert_eq!(field("rowlen=").parse::<usize>().unwrap(), c.k);
    let raw = std::fs::read(format!("{base}/ref/q5k-v41-dequant.raw")).unwrap();
    assert_eq!(
        raw.len(),
        rows * c.k * 4,
        "q5k-v41-dequant.raw: {rows} rows of {} f32",
        c.k
    );
    let want: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let mut got = vec![0.0f32; c.k];
    for r in 0..rows {
        dequant_row(GgmlType::Q5_K, c.row(r), &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(&want[r * c.k..(r + 1) * c.k]).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "row {r} value {i}: ours {g:e} (bits {:#x}) vs ggml {w:e} (bits {:#x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }
    eprintln!(
        "q5k dequant: {rows} rows x {} values bit-identical to ggml's to_float ({})",
        c.k, c.name
    );

    // A K-quant column is whole 144-byte groups: it has no tail blocks.
    assert!(
        c.acol.len().is_multiple_of(144),
        "whole 144-byte groups only"
    );
    let act = restore_q82x4(&c.acol);
    let mut max_abs = 0.0f64;
    let mut denom = 0.0f64;
    for r in 0..rows {
        let reference: f64 = want[r * c.k..(r + 1) * c.k]
            .iter()
            .zip(&act)
            .map(|(&w, &a)| f64::from(w) * a)
            .sum();
        let fused = f64::from(dot_row(GgmlType::Q5_K, c.row(r), &c.acol, c.k).unwrap());
        max_abs = max_abs.max((fused - reference).abs());
        denom = denom.max(reference.abs());
    }
    let global_rel = max_abs / denom;
    eprintln!(
        "q5k dequant band: {rows} rows, kernel vs f64 dot of ggml's rows: max|diff|={max_abs:.4e} \
         max|ref|={denom:.4e} global_rel={global_rel:.3e} (gate 2's band 1e-5)"
    );
    assert!(
        global_rel <= 1e-5,
        "kernel vs f64 dot of ggml's dequantized rows: global rel {global_rel:.3e} > 1e-5"
    );
}

// ------------------------------------------- IQ3_XXS x Q8_K and MXFP4 x Q8_2_X4
// The routed-expert formats of Qwen3.8-Flash-Next (IQ3_XXS) and V4.1 (MXFP4). The reference
// model carries neither, so each type reads the first aligned tensor of a file of its own
// ([`model_file`]): IQ3_XXS the UD-Q3_K_XL second shard's blk.0.ffn_gate_exps.weight (k = 2560),
// MXFP4 the DSpark draft's blk.0.ffn_gate_exps.weight (k = 5120). A tensor is found through the
// header-only inventory and its rows read at `data_base + offset`, as the Q5_K case does.
// Gate A: the AVX2 kernel equals its mirror bit for bit on the tensor's first rows. Gate B:
// on ik's own column the kernel is within 1 ULP of ik's kernel on all 64 dumped rows
// (tools/ref/iq3xxs_ref.cpp, tools/ref/mxfp4_x4_ref.cpp). MXFP4 also carries gate 0, the
// encoder against ik's quantize_row_q8_2_x4; IQ3_XXS cannot: ik's AVX2 q8_K encoder codes
// from the unsigned max (d > 0 always) where ours mirrors ggml's signed-extreme reference,
// so the bytes differ while the products agree.

/// The file `ty`'s gates and harness read: `BLOOMERY_IQ3XXS_MODEL` or `BLOOMERY_MXFP4_MODEL`
/// overrides the default (tools/ref/build-qdot-ref.sh reads the same variables, same
/// defaults). `BLOOMERY_V4_MODEL`, which named one file for both types, is refused when set.
fn model_file(ty: GgmlType) -> String {
    assert!(
        std::env::var_os("BLOOMERY_V4_MODEL").is_none(),
        "BLOOMERY_V4_MODEL is retired: name the IQ3_XXS file with BLOOMERY_IQ3XXS_MODEL and the \
         MXFP4 file with BLOOMERY_MXFP4_MODEL"
    );
    match ty {
        GgmlType::IQ3_XXS => std::env::var("BLOOMERY_IQ3XXS_MODEL").unwrap_or_else(|_| {
            "/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf"
                .into()
        }),
        GgmlType::MXFP4 => std::env::var("BLOOMERY_MXFP4_MODEL").unwrap_or_else(|_| {
            "/models/DeepSeek-V4.1-Flash-DSpark/DeepSeek-V4.1-Flash-Fp8-128x742M-MXFP4_MOE.tl37.gguf"
                .into()
        }),
        _ => panic!("no file case for {ty:?}"),
    }
}

/// The first `ty` tensor of [`model_file`] with an aligned k — the scan its harness
/// does — with its first rows and the column `quantize_col` codes from the first k
/// values of the attn_norm-0 dump (the column the harness codes).
struct FileCase {
    ty: GgmlType,
    name: String,
    k: usize,
    row_bytes: usize,
    bytes: Vec<u8>,
    acol: Vec<u8>,
}

impl FileCase {
    fn load(ty: GgmlType, rows: usize) -> FileCase {
        use std::os::unix::fs::FileExt;
        let (gran, block): (usize, usize) = match ty {
            GgmlType::IQ3_XXS => (256, 98),
            GgmlType::MXFP4 => (32, 17),
            _ => panic!("no file case for {ty:?}"),
        };
        let path = model_file(ty);
        let inv = gguf::inventory_of(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let t = inv
            .tensors
            .iter()
            .find(|t| t.type_id == ty.as_u32() && t.dims[0].is_multiple_of(gran as u64))
            .unwrap_or_else(|| panic!("{path} carries no aligned {ty:?} tensor"));
        let k = t.dims[0] as usize;
        let n = t.dims[1..].iter().product::<u64>() as usize;
        let row_bytes = block * k / gran;
        assert_eq!(t.nbytes, Some((n * row_bytes) as u64), "{}: bytes", t.name);
        assert!(
            n >= rows,
            "{}: only {n} rows, the gate needs {rows}",
            t.name
        );
        let mut bytes = vec![0u8; rows * row_bytes];
        std::fs::File::open(&path)
            .and_then(|f| f.read_exact_at(&mut bytes, inv.data_base + t.offset))
            .unwrap_or_else(|e| panic!("{path}: read {}: {e}", t.name));
        let xs = oracle_f32("attn_norm-0", 2048 * 6)[..k].to_vec();
        let mut acol = vec![0u8; col_bytes(ty, k)];
        quantize_col(ty, &xs, &mut acol);
        FileCase {
            ty,
            name: t.name.clone(),
            k,
            row_bytes,
            bytes,
            acol,
        }
    }

    fn row(&self, r: usize) -> &[u8] {
        &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes]
    }

    /// ik's dump for this tensor: its activation bytes and its 64 row results.
    fn ik_dump(&self, file: &str) -> (Vec<u8>, Vec<u32>) {
        let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
        let dump = std::fs::read_to_string(format!("{base}/ref/{file}"))
            .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
        let name = dump
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or_default();
        assert_eq!(
            name, self.name,
            "the dump and this scan must land on the same tensor"
        );
        let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
        assert_eq!(dump_k, self.k, "{file}: k");
        assert_eq!(
            ik_acol.len(),
            self.acol.len(),
            "ik's column must size-match ours"
        );
        (ik_acol, want)
    }

    /// Gate A: the AVX2 kernel and its mirror agree bit for bit on `ROWS` rows.
    #[cfg(target_arch = "x86_64")]
    fn gate_a(&self) {
        assert!(
            supports(self.ty),
            "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
        );
        for r in 0..ROWS {
            let a = dot_row_avx2(self.ty, self.row(r), &self.acol, self.k).unwrap();
            let b = dot_row_scalar(self.ty, self.row(r), &self.acol, self.k).unwrap();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "row {r}: kernel {a:e} (bits {:#x}) and mirror {b:e} (bits {:#x}) must be bit-identical",
                a.to_bits(),
                b.to_bits()
            );
        }
        eprintln!(
            "{:?} gate A: {ROWS} rows bit-identical (kernel vs mirror, {}, k = {})",
            self.ty, self.name, self.k
        );
    }

    /// Gate B: on ik's column, every dumped row within 1 ULP of ik's kernel.
    fn gate_b(&self, file: &str, ik_kernel: &str) {
        let (ik_acol, want) = self.ik_dump(file);
        assert!(want.len() >= 64, "dump needs 64 rows");
        let mut worst = 0i64;
        let mut exact = 0usize;
        for (r, &w) in want.iter().take(64).enumerate() {
            let got = dot_row(self.ty, self.row(r), &ik_acol, self.k).unwrap();
            let ulp = (got.to_bits() as i64 - w as i64).abs();
            assert!(
                ulp <= 1,
                "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
                f32::from_bits(w)
            );
            worst = worst.max(ulp);
            exact += usize::from(ulp == 0);
        }
        eprintln!(
            "{:?} gate B: 64 rows within 1 ULP of ik's {ik_kernel} (on ik's own activations, {}): \
             max {worst} ULP, {exact}/64 rows at 0 ULP",
            self.ty, self.name
        );
    }
}

/// IQ3_XXS gate A: kernel vs mirror on the first aligned IQ3_XXS tensor.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box, the Qwen3.8 UD-Q3_K_XL shard and $BLOOMERY_DATA/ref"]
fn hw_iq3xxs_kernel_matches_mirror() {
    FileCase::load(GgmlType::IQ3_XXS, ROWS).gate_a();
}

/// IQ3_XXS gate B: within 1 ULP of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1>`.
#[test]
#[ignore = "hw: needs the box, the Qwen3.8 UD-Q3_K_XL shard and $BLOOMERY_DATA/ref"]
fn hw_iq3xxs_kernel_predicts_ik() {
    FileCase::load(GgmlType::IQ3_XXS, 64).gate_b(
        "iq3xxs-ik-dot.txt",
        "mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1>",
    );
}

/// MXFP4 gate 0: `quantize_col` codes the column byte for byte as ik's `quantize_row_q8_2_x4`.
#[test]
#[ignore = "hw: needs the box, the DSpark draft file and $BLOOMERY_DATA/ref"]
fn hw_mxfp4_encoder_matches_ik() {
    let c = FileCase::load(GgmlType::MXFP4, 1);
    let (ik_acol, _) = c.ik_dump("mxfp4-x4-ik-dot.txt");
    for (i, (a, b)) in ik_acol.iter().zip(&c.acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!(
        "MXFP4 gate 0: {} encoder bytes bit-identical to ik's quantize_row_q8_2_x4 ({}, k = {})",
        c.acol.len(),
        c.name,
        c.k
    );
}

/// MXFP4 gate A: kernel vs mirror on the first aligned MXFP4 tensor.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box, the DSpark draft file and $BLOOMERY_DATA/ref"]
fn hw_mxfp4_kernel_matches_mirror() {
    FileCase::load(GgmlType::MXFP4, ROWS).gate_a();
}

/// MXFP4 gate B: within 1 ULP of ik's `mul_mat_qX_1_q8_2_T<MXFP4_Unpacker, 1>`.
#[test]
#[ignore = "hw: needs the box, the DSpark draft file and $BLOOMERY_DATA/ref"]
fn hw_mxfp4_kernel_predicts_ik() {
    FileCase::load(GgmlType::MXFP4, 64).gate_b(
        "mxfp4-x4-ik-dot.txt",
        "mul_mat_qX_1_q8_2_T<MXFP4_Unpacker, 1>",
    );
}

// ---------------------------------- IQ4_NL x Q8_2_X4 and IQ4_XS x Q8_K
// The Qwen3.8-Flash-Next routed-expert formats. Both gates and their harnesses read one
// dump per type (tools/ref/iq4nl_ref.cpp, tools/ref/iq4xs_ref.cpp), which carries the
// weight rows beside ik's column and results — so this side needs no model file:
//
//   clause (what the contract says)                     | assertions
//   ----------------------------------------------------|---------------------------------
//   C1 the encoder codes a column byte for byte as      | gate 0's per-byte compare (NL);
//      ik's quantize_row_q8_2_x4 (NL only: XS pairs    | col_bytes shape pins
//      q8_K, whose ik encoder differs by design)        |
//   C2 the AVX2 kernel equals its mirror bit for bit    | the dump's rows, kernel vs
//      on real rows and on generated rows at every      | mirror; ROWS generated rows at
//      tail shape                                       | each k; the s-bytes-ignored pin
//   C3 on ik's own column the kernel is within 1 ULP    | gate B: every dumped row
//      of ik's kernel (the IQ3_XXS band)                | within 1 ULP, worst reported
//   C4 the 6-bit scale split (XS) and the sign fold     | generated-row gate A at
//      (NL) decode every bit pattern                    | extreme bytes; f16 scale sweep
//
// IQ4_NL's real rows are the UD-Q4_K_XL file's per_layer_token_embd (the file this
// engine serves); IQ4_XS's are the UD-Q3_K_XL file's one IQ4_XS gate/up layer, whose
// download lands after this round — until its `.done` sentinel appears the harness
// dumps synthetic blocks under the name synthetic-iq4_xs and says so, and the gates
// run on those. A partial shard would read past EOF, so the sentinel gates the real
// path, not the file's presence.

/// One type's ik dump: the tensor name, k, ik's activation column, the weight rows the
/// kernel ran on (hex), and ik's result bits per row.
struct QwenDump {
    name: String,
    k: usize,
    ik_acol: Vec<u8>,
    rows: Vec<Vec<u8>>,
    want: Vec<u32>,
}

/// The ik commit the IQ4_XS dump is made at: `tools/ref/iq4xs-ik.pin`, the file
/// `tools/ref/build-qdot-ref.sh` builds the dump's ik tree from. The shared ik tree the other
/// dumps come from predates the signed AVX2 path this kernel follows.
const IQ4XS_IK_PIN: &str = include_str!("../../../tools/ref/iq4xs-ik.pin");

/// Whether a dump's `.meta` (its harness writes `ik=<commit>` after the dump) names `pin`.
fn dump_ik_meta_check(file: &str, meta: &str, pin: &str) -> Result<(), String> {
    match meta.lines().find_map(|l| l.strip_prefix("ik=")) {
        None => Err(format!("{file}.meta has no ik= line")),
        Some(ik) if ik == pin => Ok(()),
        Some(ik) => Err(format!("{file} was dumped by ik {ik}, the pin is {pin}")),
    }
}

/// `iq4xs-ik-dot.txt` and its `.meta` under `<base>/ref`, read as a pair: the harness holds
/// `iq4xs-ik-dot.lock` exclusive from before it drops the meta until after it writes the new one,
/// so under the shared lock the meta names the commit the dump was made at. A missing lock file or
/// meta, and a meta that names a commit other than `pin`, are refused by name.
fn read_iq4xs_pair(base: &str, pin: &str) -> Result<QwenDump, String> {
    let lock_path = format!("{base}/ref/iq4xs-ik-dot.lock");
    let lock = std::fs::File::open(&lock_path)
        .map_err(|e| format!("{lock_path}: {e} — run just build-ref-iq4xs first"))?;
    lock.lock_shared()
        .map_err(|e| format!("{lock_path}: the shared lock: {e}"))?;
    let meta_path = format!("{base}/ref/iq4xs-ik-dot.meta");
    let meta = std::fs::read_to_string(&meta_path)
        .map_err(|e| format!("{meta_path}: {e} — run just build-ref-iq4xs first"))?;
    dump_ik_meta_check("iq4xs-ik-dot", &meta, pin)
        .map_err(|why| format!("{why} — run just build-ref-iq4xs first"))?;
    Ok(qwen_dump_in(
        base,
        "iq4xs-ik-dot.txt",
        GgmlType::IQ4_XS,
        136,
        256,
    ))
}

/// The IQ4_XS dump at the pinned ik commit, or a panic naming why not.
fn iq4xs_dump() -> QwenDump {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    read_iq4xs_pair(&base, IQ4XS_IK_PIN.trim()).unwrap_or_else(|why| panic!("{why}"))
}

/// Reads `$BLOOMERY_DATA/ref/<file>`, every line by its tag; an untagged or unknown
/// line is refused, and the shapes must match the type's column and row contracts.
fn qwen_dump(file: &str, ty: GgmlType, block: usize, gran: usize) -> QwenDump {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    qwen_dump_in(&base, file, ty, block, gran)
}

/// `qwen_dump` over `<base>/ref/<file>`.
fn qwen_dump_in(base: &str, file: &str, ty: GgmlType, block: usize, gran: usize) -> QwenDump {
    let path = format!("{base}/ref/{file}");
    let dump = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e} — run just build-ref first"));
    let mut d = QwenDump {
        name: String::new(),
        k: 0,
        ik_acol: Vec::new(),
        rows: Vec::new(),
        want: Vec::new(),
    };
    for line in dump.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["tensor", name, "k", k] => {
                d.name = (*name).to_string();
                d.k = k.parse().unwrap();
            }
            [h] if h.len() > 64 => d.ik_acol = unhex(h),
            ["w", r, h] => {
                assert_eq!(r.parse::<usize>().unwrap(), d.rows.len(), "rows in order");
                d.rows.push(unhex(h));
            }
            ["row", r, bits] => {
                assert_eq!(
                    r.parse::<usize>().unwrap(),
                    d.want.len(),
                    "results in order"
                );
                d.want
                    .push(u32::from_str_radix(bits, 16).expect("ik dumps raw f32 bits"));
            }
            _ => panic!("{path}: an unknown line {:?}", &line[..line.len().min(40)]),
        }
    }
    assert!(d.k > 0 && d.k.is_multiple_of(gran), "{path}: the dump's k");
    assert_eq!(
        d.ik_acol.len(),
        col_bytes(ty, d.k),
        "{path}: ik's column must size-match ours"
    );
    assert_eq!(d.rows.len(), d.want.len(), "{path}: rows and results");
    assert!(
        d.rows.len() >= 16,
        "{path}: the dump needs at least 16 rows"
    );
    let row_bytes = block * d.k / gran;
    for (r, w) in d.rows.iter().enumerate() {
        assert_eq!(w.len(), row_bytes, "{path}: row {r}");
    }
    d
}

/// The byte offsets of a block's f16 scale fields in the generated-row gates: Q2_K carries
/// `d` and `dmin` at the block's end, every other gated type its `d` first.
fn f16_scale_fields(ty: GgmlType) -> &'static [usize] {
    match ty {
        GgmlType::Q2_K => &[80, 82],
        _ => &[0],
    }
}

/// Gate A over generated rows: `ROWS` pseudo-random rows at every `k`, every byte a
/// valid code, each block's f16 scales masked finite, against a `quantize_col` column —
/// kernel vs mirror, bit for bit.
#[cfg(target_arch = "x86_64")]
fn generated_gate_a(ty: GgmlType, block: usize, gran: usize, ks: &[usize]) {
    assert!(
        supports(ty),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let fields = f16_scale_fields(ty);
    let mut rng = Lcg(0x1AD5_0BEE_5EED);
    for &k in ks {
        let row_bytes = block * k / gran;
        let mut w = vec![0u8; ROWS * row_bytes];
        for (i, b) in w.iter_mut().enumerate() {
            *b = (rng.next_u32() >> 13) as u8;
            if fields.iter().any(|&f| i % block == f + 1) {
                *b &= 0x7b; // an f16 scale's exponent top bit: keep every scale finite
            }
        }
        let x: Vec<f32> = (0..k).map(|_| rng.unit() * 30.0).collect();
        let mut acol = vec![0u8; col_bytes(ty, k)];
        quantize_col(ty, &x, &mut acol);
        for r in 0..ROWS {
            let a = dot_row_avx2(ty, &w[r * row_bytes..], &acol, k).unwrap();
            let b = dot_row_scalar(ty, &w[r * row_bytes..], &acol, k).unwrap();
            assert!(
                a.to_bits() == b.to_bits(),
                "{ty:?} k = {k} row {r}: kernel {a:e} (bits {:#x}) and mirror {b:e} (bits {:#x}) \
                 must be bit-identical",
                a.to_bits(),
                b.to_bits()
            );
        }
        eprintln!("{ty:?} gate A: {ROWS} generated rows bit-identical at k = {k}");
    }
}

/// IQ4_NL gate 0: `quantize_col` codes the column byte for byte as ik's
/// `quantize_row_q8_2_x4` — the column the harness coded is the attn_norm dump's
/// first k values, as in the MXFP4 gate.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq4nl_encoder_matches_ik() {
    let d = qwen_dump("iq4nl-ik-dot.txt", GgmlType::IQ4_NL, 18, 32);
    let xs = oracle_f32("attn_norm-0", 2048 * 6)[..d.k].to_vec();
    let mut acol = vec![0u8; col_bytes(GgmlType::IQ4_NL, d.k)];
    quantize_col(GgmlType::IQ4_NL, &xs, &mut acol);
    for (i, (a, b)) in d.ik_acol.iter().zip(&acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    eprintln!(
        "IQ4_NL gate 0: {} encoder bytes bit-identical to ik's quantize_row_q8_2_x4 ({}, k = {})",
        acol.len(),
        d.name,
        d.k
    );
}

/// IQ4_NL gate A: kernel vs mirror on the dump's rows and on `ROWS` generated rows at
/// the down width (k = 640, whole groups) and both tail shapes — plus the pin that
/// the kernel never reads a group's i16 sum bytes (bytes 8..16): no min correction
/// exists, so corrupting them must not move the dot.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq4nl_kernel_matches_mirror() {
    assert!(
        supports(GgmlType::IQ4_NL),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let d = qwen_dump("iq4nl-ik-dot.txt", GgmlType::IQ4_NL, 18, 32);
    for r in 0..d.rows.len() {
        let a = dot_row_avx2(GgmlType::IQ4_NL, &d.rows[r], &d.ik_acol, d.k).unwrap();
        let b = dot_row_scalar(GgmlType::IQ4_NL, &d.rows[r], &d.ik_acol, d.k).unwrap();
        assert!(
            a.to_bits() == b.to_bits(),
            "dump row {r}: kernel {a:e} and mirror {b:e} must be bit-identical"
        );
    }
    eprintln!(
        "IQ4_NL gate A: {} dump rows bit-identical ({}, k = {})",
        d.rows.len(),
        d.name,
        d.k
    );
    generated_gate_a(GgmlType::IQ4_NL, 18, 32, &[640, 672, 96, 128]);

    // The s-bytes pin: a q8_2 group's i16 sums (bytes 8..16) feed only the min
    // corrections IQ4_NL does not have.
    let k = 672;
    let mut rng = Lcg(0x0FF1_600D);
    let row_bytes = 18 * k / 32;
    let mut w = vec![0u8; row_bytes];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (rng.next_u32() >> 13) as u8;
        if i % 18 == 1 {
            *b &= 0x7b;
        }
    }
    let x: Vec<f32> = (0..k).map(|_| rng.unit() * 30.0).collect();
    let mut acol = vec![0u8; col_bytes(GgmlType::IQ4_NL, k)];
    quantize_col(GgmlType::IQ4_NL, &x, &mut acol);
    let plain = dot_row(GgmlType::IQ4_NL, &w, &acol, k).unwrap();
    // Whole 144-byte groups only: the 36-byte tail block that follows them lays its
    // codes out from byte 4, and chunks_mut would hand it to this fill as a short group.
    let whole = (acol.len() / 144) * 144;
    for g in acol[..whole].chunks_mut(144) {
        g[8..16].fill(0x5a);
    }
    assert_eq!(
        dot_row(GgmlType::IQ4_NL, &w, &acol, k).unwrap().to_bits(),
        plain.to_bits(),
        "corrupting the q8_2 sums must not move an IQ4_NL dot"
    );
    // The tail's 36-byte blocks carry their sums at the same offsets.
    let tail = acol.len() - 36;
    acol[tail + 2..tail + 4].fill(0x5a);
    assert_eq!(
        dot_row(GgmlType::IQ4_NL, &w, &acol, k).unwrap().to_bits(),
        plain.to_bits(),
        "corrupting a tail block's sum must not move an IQ4_NL dot"
    );
    eprintln!("IQ4_NL gate A: the q8_2 sum bytes are never read (group and tail pinned)");
}

/// IQ4_NL gate B: every dumped row within 1 ULP of ik's
/// `mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS, 1, block_q8_2>` (the IQ3_XXS band).
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq4nl_kernel_predicts_ik() {
    let d = qwen_dump("iq4nl-ik-dot.txt", GgmlType::IQ4_NL, 18, 32);
    let mut worst = 0i64;
    let mut exact = 0usize;
    for (r, (&bits, row)) in d.want.iter().zip(&d.rows).enumerate() {
        let got = dot_row(GgmlType::IQ4_NL, row, &d.ik_acol, d.k).unwrap();
        let ulp = (got.to_bits() as i64 - bits as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(bits)
        );
        worst = worst.max(ulp);
        exact += usize::from(ulp == 0);
    }
    eprintln!(
        "IQ4_NL gate B: {} rows within 1 ULP of ik's mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS> \
         (on ik's own activations, {}): max {worst} ULP, {exact}/{} rows at 0 ULP",
        d.want.len(),
        d.name,
        d.want.len()
    );
}

/// IQ4_XS gate A: kernel vs mirror on the dump's rows (the UD-Q3_K_XL layer's rows, or
/// the harness's synthetic blocks until that file's `.done` lands) and on `ROWS`
/// generated rows at the gate/up width and both smaller block counts.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq4xs_kernel_matches_mirror() {
    assert!(
        supports(GgmlType::IQ4_XS),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let d = iq4xs_dump();
    for r in 0..d.rows.len() {
        let a = dot_row_avx2(GgmlType::IQ4_XS, &d.rows[r], &d.ik_acol, d.k).unwrap();
        let b = dot_row_scalar(GgmlType::IQ4_XS, &d.rows[r], &d.ik_acol, d.k).unwrap();
        assert!(
            a.to_bits() == b.to_bits(),
            "dump row {r}: kernel {a:e} and mirror {b:e} must be bit-identical"
        );
    }
    let origin = if d.name == "synthetic-iq4_xs" {
        "synthetic blocks — the UD-Q3_K_XL download's .done is not there yet"
    } else {
        d.name.as_str()
    };
    eprintln!(
        "IQ4_XS gate A: {} dump rows bit-identical ({}, k = {})",
        d.rows.len(),
        origin,
        d.k
    );
    generated_gate_a(GgmlType::IQ4_XS, 136, 256, &[2560, 512, 768]);
}

#[test]
fn iq4xs_signed_dot_extremes() {
    let ty = GgmlType::IQ4_XS;
    let columns: Vec<Vec<u8>> = [127i8, -127, -128]
        .into_iter()
        .map(|q| {
            let mut col = vec![0; col_bytes(ty, 256)];
            col[..4].copy_from_slice(&1.0f32.to_le_bytes());
            col[8..264].fill(q as u8);
            for sum in col[264..].as_chunks_mut::<2>().0 {
                sum.copy_from_slice(&(16 * i16::from(q)).to_le_bytes());
            }
            col
        })
        .collect();
    for (nibble, want) in [
        (0xff, [3_673_856.0f32, -3_673_856.0, -3_702_784.0]),
        (0x00, [-4_129_024.0f32, 4_129_024.0, -4_161_536.0]),
    ] {
        for scale in [-32i16, 0, 1, 31] {
            let code = (scale + 32) as u16;
            let row = iq4xs_block(
                0x3c00,
                (code >> 4) * 0x5555,
                [((code & 15) * 17) as u8; 4],
                |_| nibble,
            );
            let expected = want.map(|v| {
                if scale == 0 {
                    0.0
                } else {
                    v * f32::from(scale)
                }
            });
            for (col, want) in columns.iter().zip(expected) {
                assert_eq!(
                    dot_row_scalar(ty, &row, col, 256).unwrap().to_bits(),
                    want.to_bits()
                );
                assert_eq!(
                    dot_row(ty, &row, col, 256).unwrap().to_bits(),
                    want.to_bits()
                );
            }
            for c in 1..=qdot::TILE_COLS {
                let acols: Vec<&[u8]> = (0..c).map(|i| columns[i % 3].as_slice()).collect();
                let mut out = vec![0.0; c];
                qdot::dot_row_cols(ty, &row, &acols, 256, &mut out).unwrap();
                for (i, got) in out.iter().enumerate() {
                    assert_eq!(got.to_bits(), expected[i % 3].to_bits());
                }
            }
        }
    }
}

/// The IQ4_XS dump is read only at the pinned ik commit: the pin file is one full commit, and a
/// meta without an `ik=` line, naming another commit or only an abbreviation of the pin is
/// refused by name.
#[test]
fn iq4xs_dump_meta_names_the_pin() {
    let pin = IQ4XS_IK_PIN.trim();
    assert!(
        pin.len() == 40
            && pin
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "tools/ref/iq4xs-ik.pin must hold one 40-digit lower-case commit, not {pin:?}"
    );
    let ours = format!("ik={pin}\nbuilt=2026-10-21T00:00:00Z\n");
    assert_eq!(dump_ik_meta_check("d", &ours, pin), Ok(()));
    let stale = "ik=c10fbbcccd69cb1c66ddeb39601b0cd71701f7c3\nbuilt=2026-10-20T00:00:00Z\n";
    let err = dump_ik_meta_check("d", stale, pin).unwrap_err();
    assert!(
        err.contains("c10fbbcccd69cb1c66ddeb39601b0cd71701f7c3") && err.contains(pin),
        "{err}"
    );
    assert!(dump_ik_meta_check("d", &format!("ik={}\n", &pin[..8]), pin).is_err());
    let err = dump_ik_meta_check("d", "built=2026-10-21T00:00:00Z\n", pin).unwrap_err();
    assert_eq!(err, "d.meta has no ik= line");
}

/// A scratch `$BLOOMERY_DATA` (`ref/` inside), removed on drop.
struct ScratchData(std::path::PathBuf);

impl ScratchData {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("qdot-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("ref")).expect("scratch data dir");
        Self(dir)
    }

    fn base(&self) -> &str {
        self.0.to_str().expect("utf-8 temp dir")
    }

    fn file(&self, name: &str) -> String {
        format!("{}/ref/{name}", self.base())
    }
}

impl Drop for ScratchData {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A dump of the IQ4_XS contract's smallest shape (k = 256, 16 rows) with zero blocks.
fn iq4xs_dump_text() -> String {
    let mut s = format!(
        "tensor t k 256\n{}\n",
        "00".repeat(col_bytes(GgmlType::IQ4_XS, 256))
    );
    for r in 0..16 {
        s += &format!("w {r} {}\n", "00".repeat(136));
    }
    for r in 0..16 {
        s += &format!("row {r} 00000000\n");
    }
    s
}

/// The pair reader refuses by name, before it reads the dump: a missing lock file or meta, a meta
/// without an `ik=` line, and a meta naming another commit. A pair at the pin loads.
#[test]
fn iq4xs_pair_reader_refuses_by_name() {
    let pin = IQ4XS_IK_PIN.trim();
    let s = ScratchData::new("pair-refuse");
    let err = read_iq4xs_pair(s.base(), pin).err().expect("no lock file");
    assert!(
        err.contains("iq4xs-ik-dot.lock") && err.contains("just build-ref-iq4xs"),
        "{err}"
    );
    std::fs::write(s.file("iq4xs-ik-dot.lock"), "").unwrap();
    let err = read_iq4xs_pair(s.base(), pin).err().expect("no meta");
    assert!(err.contains("iq4xs-ik-dot.meta"), "{err}");
    std::fs::write(s.file("iq4xs-ik-dot.meta"), "built=2026-10-21T00:00:00Z\n").unwrap();
    let err = read_iq4xs_pair(s.base(), pin).err().expect("no ik= line");
    assert!(err.contains("has no ik= line"), "{err}");
    let stale = "c10fbbcccd69cb1c66ddeb39601b0cd71701f7c3";
    std::fs::write(s.file("iq4xs-ik-dot.meta"), format!("ik={stale}\n")).unwrap();
    let err = read_iq4xs_pair(s.base(), pin).err().expect("stale meta");
    assert!(err.contains(stale) && err.contains(pin), "{err}");
    std::fs::write(s.file("iq4xs-ik-dot.meta"), format!("ik={pin}\n")).unwrap();
    std::fs::write(s.file("iq4xs-ik-dot.txt"), iq4xs_dump_text()).unwrap();
    let d = read_iq4xs_pair(s.base(), pin).expect("a pair at the pin");
    assert_eq!((d.k, d.rows.len(), d.want.len()), (256, 16, 16));
}

/// The pair reader reads nothing while the harness holds the pair's lock, and reads the new pair
/// once it lets go.
#[test]
fn iq4xs_pair_reader_waits_for_the_writer() {
    use std::time::Duration;
    let pin = IQ4XS_IK_PIN.trim().to_string();
    let s = ScratchData::new("pair-wait");
    let lock_path = s.file("iq4xs-ik-dot.lock");
    std::fs::write(s.file("iq4xs-ik-dot.meta"), format!("ik={pin}\n")).unwrap();
    std::fs::write(s.file("iq4xs-ik-dot.txt"), iq4xs_dump_text()).unwrap();
    let writer = std::fs::File::create(&lock_path).unwrap();
    writer.lock().unwrap();
    let base = s.base().to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        tx.send(read_iq4xs_pair(&base, &pin).map(|d| d.k)).unwrap();
    });
    let early = rx.recv_timeout(Duration::from_millis(500));
    writer.unlock().unwrap();
    let late = rx.recv_timeout(Duration::from_secs(30));
    reader.join().unwrap();
    assert!(
        early.is_err(),
        "the reader returned {early:?} while the writer held the lock"
    );
    assert_eq!(late.expect("the reader after the unlock"), Ok(256));
}

/// A scratch data dir for the built `iq4xs_ref` harness: the oracle's activation dump copied in,
/// and the model path of a file that is not there, which makes the harness dump synthetic blocks.
fn iq4xs_ref_scratch(tag: &str) -> (ScratchData, std::process::Command) {
    let real = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let bin = format!("{real}/bin/iq4xs_ref");
    assert!(
        std::path::Path::new(&bin).exists(),
        "{bin}: run just build-ref-iq4xs first"
    );
    let s = ScratchData::new(tag);
    let norm = "attn_norm-0.0.f32";
    std::fs::copy(format!("{real}/ref/{norm}"), s.file(norm))
        .expect("the oracle's activation dump");
    let mut cmd = std::process::Command::new(bin);
    cmd.arg(s.file("no-model.gguf"))
        .env("BLOOMERY_DATA", s.base())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    (s, cmd)
}

/// The harness takes the pair's lock exclusive and drops the meta before it publishes the dump,
/// so a meta write that fails leaves a dump with no meta, never a meta of an earlier commit
/// beside a dump of this one. Its meta names the commit of `tools/ref/iq4xs-ik.pin`.
#[test]
#[ignore = "hw: needs the box and the built iq4xs_ref"]
fn hw_iq4xs_ref_invalidates_the_meta_before_it_publishes() {
    let pin = IQ4XS_IK_PIN.trim();
    let (s, mut cmd) = iq4xs_ref_scratch("ref-invalidate");
    assert!(cmd.status().unwrap().success(), "the harness run");
    let meta = std::fs::read_to_string(s.file("iq4xs-ik-dot.meta")).expect("the harness's meta");
    dump_ik_meta_check("iq4xs-ik-dot", &meta, pin)
        .unwrap_or_else(|why| panic!("{why} — the harness binary is not the pin's: rebuild it"));
    read_iq4xs_pair(s.base(), pin).expect("a pair at the pin");
    // A meta of another commit stands; the next run's meta write fails (its temporary name is a
    // directory: `sh` execs the harness, so the pid in the name is the harness's).
    let earlier = "c10fbbcccd69cb1c66ddeb39601b0cd71701f7c3";
    std::fs::write(s.file("iq4xs-ik-dot.meta"), format!("ik={earlier}\n")).unwrap();
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(r#"mkdir "$1.tmp.$$" && exec "$2" "$3""#)
        .args(["sh", &s.file("iq4xs-ik-dot.meta")])
        .arg(cmd.get_program())
        .arg(s.file("no-model.gguf"))
        .env("BLOOMERY_DATA", s.base())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "the harness run with a failing meta write"
    );
    assert!(
        !std::path::Path::new(&s.file("iq4xs-ik-dot.meta")).exists(),
        "a meta of ik {earlier} is left beside the dump of the run that failed to write its own"
    );
    let err = read_iq4xs_pair(s.base(), pin)
        .err()
        .expect("a dump with no meta");
    assert!(err.contains("iq4xs-ik-dot.meta"), "{err}");
}

/// The harness publishes nothing while a reader holds the pair's shared lock, and publishes once
/// it lets go.
#[test]
#[ignore = "hw: needs the box and the built iq4xs_ref"]
fn hw_iq4xs_ref_publishes_under_the_pair_lock() {
    use std::time::{Duration, Instant};
    let pin = IQ4XS_IK_PIN.trim();
    let (s, mut cmd) = iq4xs_ref_scratch("ref-lock");
    let reader = std::fs::File::create(s.file("iq4xs-ik-dot.lock")).unwrap();
    reader.lock_shared().unwrap();
    let mut child = cmd.spawn().unwrap();
    std::thread::sleep(Duration::from_secs(1));
    let waiting = child.try_wait().unwrap().is_none();
    let published = ["iq4xs-ik-dot.txt", "iq4xs-ik-dot.meta"]
        .iter()
        .any(|f| std::path::Path::new(&s.file(f)).exists());
    reader.unlock().unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            panic!("the harness did not finish within 60 s of the unlock");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(waiting, "the harness finished while a reader held the lock");
    assert!(
        !published,
        "the harness published while a reader held the lock"
    );
    assert!(status.success(), "the harness run after the unlock");
    read_iq4xs_pair(s.base(), pin).expect("the pair the harness published");
}

/// IQ4_XS gate B: every dumped row within 1 ULP of ik's
/// `mul_mat_qX_K_q8_K_T<DequantizerIQ4XS, 1>` (the IQ3_XXS band).
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq4xs_kernel_predicts_ik() {
    let d = iq4xs_dump();
    let mut worst = 0i64;
    let mut exact = 0usize;
    for (r, (&bits, row)) in d.want.iter().zip(&d.rows).enumerate() {
        let got = dot_row(GgmlType::IQ4_XS, row, &d.ik_acol, d.k).unwrap();
        let ulp = (got.to_bits() as i64 - bits as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(bits)
        );
        worst = worst.max(ulp);
        exact += usize::from(ulp == 0);
    }
    let origin = if d.name == "synthetic-iq4_xs" {
        "synthetic blocks"
    } else {
        d.name.as_str()
    };
    eprintln!(
        "IQ4_XS gate B: {} rows within 1 ULP of ik's mul_mat_qX_K_q8_K_T<DequantizerIQ4XS> \
         (on ik's own activations, {}): max {worst} ULP, {exact}/{} rows at 0 ULP",
        d.want.len(),
        origin,
        d.want.len()
    );
}

// ------------------------------------------------------- IQ3_S x Q8_K
// No IQ3_S file is on the box, so both gates read the dump tools/ref/iq3s_ref.cpp writes: 64
// synthetic rows at k = 4096 (32 ggml-quantized, 32 of random codes that reach every grid
// index and sign byte) beside ik's column and results, under the name synthetic-iq3_s.

/// An IQ3_S block: f16 `d`, the 64 index bytes from `qs`, the 8 high-bit bytes from `qh`, the
/// 32 sign bytes from `signs` and the 4 scale bytes.
fn iq3s_block(
    d: u16,
    qs: impl Fn(usize) -> u8,
    qh: impl Fn(usize) -> u8,
    signs: impl Fn(usize) -> u8,
    scales: [u8; 4],
) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    b.extend((0..64).map(qs));
    b.extend((0..8).map(qh));
    b.extend((0..32).map(signs));
    b.extend_from_slice(&scales);
    b
}

/// IQ3_S end blocks: every index bit set (grid entry 511) with every sign and the top
/// scales under either sign of `d`, the same with no signs, the smallest scales, every sign
/// byte alone (the `qh` bits clear), every `qh` bit alone (the indices 256 + 0), a scrambled
/// pattern, d = 0 and a subnormal d.
const IQ3S_ENDS: [IqEnd; 9] = [
    || iq3s_block(F16_ONE, |_| 0xFF, |_| 0xFF, |_| 0xFF, [0xFF; 4]),
    || iq3s_block(F16_NEG_ONE, |_| 0xFF, |_| 0xFF, |_| 0xFF, [0xFF; 4]),
    || iq3s_block(F16_ONE, |_| 0xFF, |_| 0xFF, |_| 0, [0xFF; 4]),
    || iq3s_block(F16_ONE, |_| 0xFF, |_| 0xFF, |_| 0xFF, [0; 4]),
    || iq3s_block(F16_ONE, |i| i as u8, |_| 0, |_| 0xFF, [0xFF; 4]),
    || iq3s_block(F16_ONE, |_| 0, |_| 0xFF, |_| 0, [0xFF; 4]),
    || {
        iq3s_block(
            F16_ONE,
            |i| (i * 37 + 11) as u8,
            |i| (0xA5C3u16.rotate_left(3 * i as u32) & 0xFF) as u8,
            |i| (i * 53 + 7) as u8,
            [0x5A, 0xA5, 0x3C, 0xC3],
        )
    },
    || iq3s_block(0, |_| 0xFF, |_| 0xFF, |_| 0xFF, [0xFF; 4]),
    || iq3s_block(F16_TINY, |i| i as u8, |_| 0xAA, |_| 0xF0, [0xF0; 4]),
];

/// IQ3_S gate A: kernel vs mirror on the dump's synthetic rows, on `ROWS` generated rows at
/// the gate/up width and both smaller block counts, and on the ends of [`IQ3S_ENDS`] (every
/// index bit, sign bit and scale at its extreme, d = 0, a subnormal d) against the ±3.0 and
/// seeded columns — bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq3s_kernel_matches_mirror() {
    assert!(
        supports(GgmlType::IQ3_S),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let d = qwen_dump("iq3s-ik-dot.txt", GgmlType::IQ3_S, 110, 256);
    for r in 0..d.rows.len() {
        let a = dot_row_avx2(GgmlType::IQ3_S, &d.rows[r], &d.ik_acol, d.k).unwrap();
        let b = dot_row_scalar(GgmlType::IQ3_S, &d.rows[r], &d.ik_acol, d.k).unwrap();
        assert!(
            a.to_bits() == b.to_bits(),
            "dump row {r}: kernel {a:e} and mirror {b:e} must be bit-identical"
        );
    }
    eprintln!(
        "IQ3_S gate A: {} dump rows bit-identical ({}, k = {})",
        d.rows.len(),
        d.name,
        d.k
    );
    generated_gate_a(GgmlType::IQ3_S, 110, 256, &[4096, 512, 768]);
    for k in [4096, 256] {
        let nb = k / 256;
        let rows = legacy_end_rows(&IQ3S_ENDS, nb, |end| end());
        let cols = legacy_tile_columns(GgmlType::IQ3_S, k, vec![]);
        for (r, row) in rows.chunks(110 * nb).enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let a = dot_row_avx2(GgmlType::IQ3_S, row, col, k).unwrap();
                let b = dot_row_scalar(GgmlType::IQ3_S, row, col, k).unwrap();
                assert!(
                    a.to_bits() == b.to_bits(),
                    "end row {r} (k = {k}) column {j}: kernel {a:e} (bits {:#x}) and mirror {b:e} \
                     (bits {:#x}) must be bit-identical",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        eprintln!(
            "IQ3_S gate A: {} end rows x {} columns bit-identical (k = {k})",
            rows.len() / (110 * nb),
            cols.len()
        );
    }
}

/// IQ3_S gate B: every dumped row within 1 ULP of ik's
/// `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3S, 1>` (the IQ3_XXS band; 0 ULP is the target), the
/// worst row reported.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq3s_kernel_predicts_ik() {
    let d = qwen_dump("iq3s-ik-dot.txt", GgmlType::IQ3_S, 110, 256);
    let mut worst = (0i64, 0usize);
    let mut exact = 0usize;
    for (r, (&bits, row)) in d.want.iter().zip(&d.rows).enumerate() {
        let got = dot_row(GgmlType::IQ3_S, row, &d.ik_acol, d.k).unwrap();
        let ulp = (got.to_bits() as i64 - bits as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(bits)
        );
        if ulp > worst.0 {
            worst = (ulp, r);
        }
        exact += usize::from(ulp == 0);
    }
    let at = if worst.0 > 0 {
        format!(" (row {})", worst.1)
    } else {
        String::new()
    };
    eprintln!(
        "IQ3_S gate B: {} rows within 1 ULP of ik's mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3S, 1> \
         (on ik's own activations, {}): max {} ULP{at}, {exact}/{} rows at 0 ULP",
        d.want.len(),
        d.name,
        worst.0,
        d.want.len()
    );
}

// ------------------------------------------- Q2_K and IQ2_XS x Q8_K
// GLM-5.3-Flash's UD-Q2_K_XL routed experts. Both gates read the dump tools/ref/q2k_ref.cpp or
// tools/ref/iq2xs_ref.cpp writes: 96 rows at one k, the first 64 real rows of the file's first
// aligned tensor of the type (the harness names it in the dump), the last 32 synthetic rows of
// random codes (every scale and min nibble, grid index and sign index), beside ik's column and
// results. There is no per-row marker; the split is the rows' order.

/// The dump's real rows, which come first.
const Q2_DUMP_REAL: usize = 64;
/// The dump's rows in all: the real ones, then the synthetic random-code rows.
const Q2_DUMP_ROWS: usize = 96;

/// Reads one of the two dumps and checks it holds the harness's 64 real and 32 synthetic rows.
fn glm_q2_dump(file: &str, ty: GgmlType, block: usize) -> QwenDump {
    let d = qwen_dump(file, ty, block, 256);
    assert_eq!(
        d.rows.len(),
        Q2_DUMP_ROWS,
        "{file}: the harness writes {Q2_DUMP_REAL} real and {} synthetic rows",
        Q2_DUMP_ROWS - Q2_DUMP_REAL
    );
    d
}

/// A Q2_K block: the 16 scale bytes from `scales` (scale in the low nibble, min in the high),
/// the 64 code bytes from `qs`, then the f16 `d` and `dmin`.
fn q2k_block(scales: impl Fn(usize) -> u8, qs: impl Fn(usize) -> u8, d: u16, dmin: u16) -> Vec<u8> {
    let mut b: Vec<u8> = (0..16).map(scales).collect();
    b.extend((0..64).map(qs));
    b.extend_from_slice(&d.to_le_bytes());
    b.extend_from_slice(&dmin.to_le_bytes());
    b
}

/// An IQ2_XS block: f16 `d`, the 32 u16 codes from `codes` (9-bit grid index, 7-bit sign index
/// above it) and the 8 scale bytes (two 4-bit scales a byte).
fn iq2xs_block(d: u16, codes: impl Fn(usize) -> u16, scales: [u8; 8]) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    for i in 0..32 {
        b.extend_from_slice(&codes(i).to_le_bytes());
    }
    b.extend_from_slice(&scales);
    b
}

/// Q2_K end blocks: every scale and min 15 with every code 3 under either sign of `d` and
/// `dmin`, the scales alone (mins 0), the mins alone (scales 0), every code 0, a scrambled
/// pattern, `d` = `dmin` = 0, a subnormal `d` beside a unit `dmin`, and `dmin` against the sign
/// of `d`.
const Q2K_ENDS: [IqEnd; 9] = [
    || q2k_block(|_| 0xFF, |_| 0xFF, F16_ONE, F16_ONE),
    || q2k_block(|_| 0xFF, |_| 0xFF, F16_NEG_ONE, F16_NEG_ONE),
    || q2k_block(|_| 0x0F, |_| 0xFF, F16_ONE, F16_ONE),
    || q2k_block(|_| 0xF0, |_| 0xFF, F16_ONE, F16_ONE),
    || q2k_block(|_| 0xFF, |_| 0, F16_ONE, F16_ONE),
    || {
        q2k_block(
            |i| (i * 37 + 11) as u8,
            |i| (i * 53 + 7) as u8,
            F16_ONE,
            F16_ONE,
        )
    },
    || q2k_block(|_| 0xFF, |_| 0xFF, 0, 0),
    || q2k_block(|_| 0xFF, |i| (i * 37) as u8, F16_TINY, F16_ONE),
    || q2k_block(|_| 0xFF, |_| 0xFF, F16_ONE, F16_NEG_ONE),
];

/// IQ2_XS end blocks: every code 0xFFFF (grid entry 511 under sign index 127) with the top
/// scales under either sign of `d`, a scrambled pattern, the sign index alone (grid index 0),
/// the grid index alone (sign index 0), the smallest scales, d = 0 and a subnormal d.
const IQ2XS_ENDS: [IqEnd; 8] = [
    || iq2xs_block(F16_ONE, |_| 0xFFFF, [0xFF; 8]),
    || iq2xs_block(F16_NEG_ONE, |_| 0xFFFF, [0xFF; 8]),
    || {
        iq2xs_block(
            F16_ONE,
            |i| (i as u16).wrapping_mul(37).wrapping_add(11),
            [0x5A, 0xA5, 0x3C, 0xC3, 0x96, 0x69, 0x1E, 0xE1],
        )
    },
    || iq2xs_block(F16_ONE, |_| 0xFE00, [0xFF; 8]),
    || iq2xs_block(F16_ONE, |_| 0x01FF, [0xFF; 8]),
    || iq2xs_block(F16_ONE, |_| 0xFFFF, [0; 8]),
    || iq2xs_block(0, |_| 0xFFFF, [0xFF; 8]),
    || iq2xs_block(F16_TINY, |i| i as u16 * 0x0409, [0xF0; 8]),
];

/// Q2_K gate A: kernel vs mirror on the dump's rows, on `ROWS` generated rows at the dump's
/// width and both smaller block counts, and on the ends of [`Q2K_ENDS`] (every scale, min and
/// code at its extreme, `d` and `dmin` of either sign, zero and subnormal) against the ±3.0
/// and seeded columns — bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_q2k_kernel_matches_mirror() {
    assert!(
        supports(GgmlType::Q2_K),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let d = glm_q2_dump("q2k-ik-dot.txt", GgmlType::Q2_K, 84);
    for r in 0..d.rows.len() {
        let a = dot_row_avx2(GgmlType::Q2_K, &d.rows[r], &d.ik_acol, d.k).unwrap();
        let b = dot_row_scalar(GgmlType::Q2_K, &d.rows[r], &d.ik_acol, d.k).unwrap();
        assert!(
            a.to_bits() == b.to_bits(),
            "dump row {r}: kernel {a:e} and mirror {b:e} must be bit-identical"
        );
    }
    eprintln!(
        "Q2_K gate A: {} dump rows bit-identical ({}, k = {})",
        d.rows.len(),
        d.name,
        d.k
    );
    generated_gate_a(GgmlType::Q2_K, 84, 256, &[4096, 512, 768]);
    for k in [4096, 256] {
        let nb = k / 256;
        let rows = legacy_end_rows(&Q2K_ENDS, nb, |end| end());
        let cols = legacy_tile_columns(GgmlType::Q2_K, k, vec![]);
        for (r, row) in rows.chunks(84 * nb).enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let a = dot_row_avx2(GgmlType::Q2_K, row, col, k).unwrap();
                let b = dot_row_scalar(GgmlType::Q2_K, row, col, k).unwrap();
                assert!(
                    a.to_bits() == b.to_bits(),
                    "end row {r} (k = {k}) column {j}: kernel {a:e} (bits {:#x}) and mirror {b:e} \
                     (bits {:#x}) must be bit-identical",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        eprintln!(
            "Q2_K gate A: {} end rows x {} columns bit-identical (k = {k})",
            rows.len() / (84 * nb),
            cols.len()
        );
    }
}

/// Q2_K gate B: every dumped row bit for bit equal to ik's
/// `mul_mat_qY_K_q8_K_T<DequantizerQ2K, 1>` (iqk_gemm_kquants.cpp:678) on ik's own column.
/// Both run the same integer sums (exact in i32) and the same float steps in the same order
/// (the fmadd of the bsums fold, the fmadd of the codes, `hsum_float_8`), so any difference is
/// a bug, not rounding.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_q2k_kernel_predicts_ik() {
    let d = glm_q2_dump("q2k-ik-dot.txt", GgmlType::Q2_K, 84);
    let mut worst = 0i64;
    for (r, (&bits, row)) in d.want.iter().zip(&d.rows).enumerate() {
        let got = dot_row(GgmlType::Q2_K, row, &d.ik_acol, d.k).unwrap();
        let ulp = (got.to_bits() as i64 - bits as i64).abs();
        assert!(
            ulp == 0,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(bits)
        );
        worst = worst.max(ulp);
    }
    eprintln!(
        "Q2_K gate B: {} rows bit-identical to ik's mul_mat_qY_K_q8_K_T<DequantizerQ2K, 1> (on \
         ik's own activations, {}; the first {Q2_DUMP_REAL} rows real, the last {} synthetic): \
         max {} ULP",
        d.want.len(),
        d.name,
        d.want.len() - Q2_DUMP_REAL,
        worst
    );
}

/// IQ2_XS gate A: kernel vs mirror on the dump's rows, on `ROWS` generated rows at the dump's
/// width and both smaller block counts, and on the ends of [`IQ2XS_ENDS`] (every grid index,
/// sign index and scale at its extreme, d = 0, a subnormal d) against the ±3.0 and seeded
/// columns — bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq2xs_kernel_matches_mirror() {
    assert!(
        supports(GgmlType::IQ2_XS),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let d = glm_q2_dump("iq2xs-ik-dot.txt", GgmlType::IQ2_XS, 74);
    for r in 0..d.rows.len() {
        let a = dot_row_avx2(GgmlType::IQ2_XS, &d.rows[r], &d.ik_acol, d.k).unwrap();
        let b = dot_row_scalar(GgmlType::IQ2_XS, &d.rows[r], &d.ik_acol, d.k).unwrap();
        assert!(
            a.to_bits() == b.to_bits(),
            "dump row {r}: kernel {a:e} and mirror {b:e} must be bit-identical"
        );
    }
    eprintln!(
        "IQ2_XS gate A: {} dump rows bit-identical ({}, k = {})",
        d.rows.len(),
        d.name,
        d.k
    );
    generated_gate_a(GgmlType::IQ2_XS, 74, 256, &[4096, 512, 768]);
    for k in [4096, 256] {
        let nb = k / 256;
        let rows = legacy_end_rows(&IQ2XS_ENDS, nb, |end| end());
        let cols = legacy_tile_columns(GgmlType::IQ2_XS, k, vec![]);
        for (r, row) in rows.chunks(74 * nb).enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let a = dot_row_avx2(GgmlType::IQ2_XS, row, col, k).unwrap();
                let b = dot_row_scalar(GgmlType::IQ2_XS, row, col, k).unwrap();
                assert!(
                    a.to_bits() == b.to_bits(),
                    "end row {r} (k = {k}) column {j}: kernel {a:e} (bits {:#x}) and mirror {b:e} \
                     (bits {:#x}) must be bit-identical",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        eprintln!(
            "IQ2_XS gate A: {} end rows x {} columns bit-identical (k = {k})",
            rows.len() / (74 * nb),
            cols.len()
        );
    }
}

/// IQ2_XS gate B: every dumped row bit for bit equal to ik's
/// `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ2XS, 1>` (iqk_gemm_iquants.cpp:788) on ik's own column.
/// Both run the same integer sums (exact in i32) and the same float steps in the same order
/// (the fmadd of the bsums fold, the fmadd of the codes, `hsum_float_8`), so any difference is
/// a bug, not rounding.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_iq2xs_kernel_predicts_ik() {
    let d = glm_q2_dump("iq2xs-ik-dot.txt", GgmlType::IQ2_XS, 74);
    let mut worst = 0i64;
    for (r, (&bits, row)) in d.want.iter().zip(&d.rows).enumerate() {
        let got = dot_row(GgmlType::IQ2_XS, row, &d.ik_acol, d.k).unwrap();
        let ulp = (got.to_bits() as i64 - bits as i64).abs();
        assert!(
            ulp == 0,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(bits)
        );
        worst = worst.max(ulp);
    }
    eprintln!(
        "IQ2_XS gate B: {} rows bit-identical to ik's mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ2XS, 1> \
         (on ik's own activations, {}; the first {Q2_DUMP_REAL} rows real, the last {} \
         synthetic): max {} ULP",
        d.want.len(),
        d.name,
        d.want.len() - Q2_DUMP_REAL,
        worst
    );
}

// ------------------------------------------------------- IQ2_XXS x Q8_K
// The kernel is ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ2XXS, nrc_y>` and no harness dumps
// it, so the gates stand on the crate's own oracles: kernel vs mirror (bit identity, the twins'
// contract), ours vs the dequant oracle (a band, below) and tile vs dot_row (bit identity).

/// An IQ2_XXS block: f16 `d`, then per sub-block `s` the four grid index bytes `idx(4s)`
/// .. `idx(4s + 3)` and the sign/scale word `aux(s)` (bits `7l .. 7l + 7` group `l`'s
/// [`KSIGNS_IQ2XS`] index, bits 28..32 the scale nibble).
fn iq2xxs_block(d: u16, idx: impl Fn(usize) -> u8, aux: impl Fn(usize) -> u32) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    for s in 0..8 {
        b.extend((0..4).map(|l| idx(4 * s + l)));
        b.extend_from_slice(&aux(s).to_le_bytes());
    }
    b
}

/// The band gate's hand-built block: sub-block `s` reads grid entry `s` in all four
/// groups, sign index `7 − s` on every group and the scale nibble `s`.
fn iq2xxs_hand_block() -> Vec<u8> {
    iq2xxs_block(
        F16_ONE,
        |i| ((i / 4) % 8) as u8,
        |s| {
            let v = 7 - s as u32;
            (s as u32) << 28 | v | v << 7 | v << 14 | v << 21
        },
    )
}

/// IQ2_XXS end blocks: every index 0xFF with every sign bit and the top scale under
/// either sign of `d`, the signs alone (scale 0), the scale alone (no signs), every index
/// 0 under a scrambled word, a scrambled pattern, d = 0 and a subnormal d.
const IQ2XXS_ENDS: [IqEnd; 8] = [
    || iq2xxs_block(F16_ONE, |_| 0xFF, |_| 0xFFFF_FFFF),
    || iq2xxs_block(F16_NEG_ONE, |_| 0xFF, |_| 0xFFFF_FFFF),
    || iq2xxs_block(F16_ONE, |_| 0xFF, |_| 0x0FFF_FFFF),
    || iq2xxs_block(F16_ONE, |_| 0xFF, |_| 0xF000_0000),
    || iq2xxs_block(F16_ONE, |_| 0, |_| 0xA5C3_1E87),
    || {
        iq2xxs_block(
            F16_ONE,
            |i| (i * 37 + 11) as u8,
            |s| 0x1E87_A5C3u32.rotate_left(5 * s as u32),
        )
    },
    || iq2xxs_block(0, |_| 0xFF, |_| 0xFFFF_FFFF),
    || iq2xxs_block(F16_TINY, |i| i as u8, |_| 0xF0F0_F0F0),
];

/// IQ2_XXS kernel vs mirror on [`generated_gate_a`]'s rows (random bytes, finite scales)
/// at 16, three and one blocks, and on the ends of [`IQ2XXS_ENDS`] (every grid index, sign
/// index and scale nibble at its extreme, d = 0, a subnormal d) at one and three blocks
/// against the ±3.0 and seeded columns — bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq2xxs_kernel_matches_mirror() {
    generated_gate_a(GgmlType::IQ2_XXS, 66, 256, &[4096, 768, 256]);
    for k in [256, 768] {
        let nb = k / 256;
        let rows = legacy_end_rows(&IQ2XXS_ENDS, nb, |end| end());
        let cols = legacy_tile_columns(GgmlType::IQ2_XXS, k, vec![]);
        for (r, row) in rows.chunks(66 * nb).enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let a = dot_row_avx2(GgmlType::IQ2_XXS, row, col, k).unwrap();
                let b = dot_row_scalar(GgmlType::IQ2_XXS, row, col, k).unwrap();
                assert!(
                    a.to_bits() == b.to_bits(),
                    "end row {r} (k = {k}) column {j}: kernel {a:e} (bits {:#x}) and mirror {b:e} \
                     (bits {:#x}) must be bit-identical",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        eprintln!(
            "IQ2_XXS: {} end rows x {} columns bit-identical (k = {k})",
            rows.len() / (66 * nb),
            cols.len()
        );
    }
}

/// IQ2_XXS against the dequant oracle: `dequant_row`'s f32 weights (held to ggml's
/// `dequantize_row_iq2_xxs` bit for bit in gguf) times the column's scaled codes, summed
/// exactly in f64, at the hand-built block ([`iq2xxs_hand_block`]), the ends' one- and
/// three-block rows and sixteen random blocks, through `dot_row` and `dot_row_scalar`.
///
/// The oracle's weights carry two f32 roundings a term (`db`, then the value). Ours sums
/// exact integer lanes in two fmadds a block and lane — the bsums fold's against the
/// rounded `-43·d·dy`, the codes' against the rounded `d·dy` — then the closing hsum's three
/// adds; `d = 0.125·f16` and `43·d` are exact. A lane's codes sum carries each value's code
/// times `|grid| + 43` and its fold takes `43·Σ code` of one sub-block back out, so the sums
/// those roundings act on are bounded by `Σ 0.125·|d|·(2s + 1)·(|grid| + 86)·|dy·q|`, that is
/// `Σ (|w| + 0.125·|d|·(2s + 1)·86)·|dy·q|` over the values. The band is
/// `gamma_(2nb + 10)` of that sum, with `gamma_n = n·u / (1 − n·u)` and `u = 2^-24`: both
/// sides' roundings over the exact dot. A wrong grid byte, sign bit, scale nibble or fold
/// term moves a whole term, far past it.
#[test]
fn iq2xxs_dot_is_within_its_f64_bound() {
    let mut rng = Lcg(0x2C_0F7E);
    for k in [256usize, 768, 4096] {
        let nb = k / 256;
        let x: Vec<f32> = (0..k).map(|_| rng.unit() * 30.0).collect();
        let mut acol = vec![0u8; col_bytes(GgmlType::IQ2_XXS, k)];
        quantize_col(GgmlType::IQ2_XXS, &x, &mut acol);
        let mut rows = vec![iq2xxs_hand_block().repeat(nb)];
        rows.extend(
            legacy_end_rows(&IQ2XXS_ENDS, nb, |end| end())
                .chunks(66 * nb)
                .map(<[u8]>::to_vec),
        );
        for _ in 0..4 {
            rows.push(iq_random_block(&mut rng, 66, &[0]).repeat(nb));
        }
        // PIN(2026-10-23): gamma_(2nb + 10) over the offset sums, was gamma_(nb + 10) over Σ|term|: ik's form (hostsimd3) adds the fold's fmadd a block and lane, and its lanes carry the ±43 offset.
        let n_u = (2 * nb + 10) as f64 * 2f64.powi(-24);
        for (r, row) in rows.iter().enumerate() {
            let mut w = vec![0.0f32; k];
            dequant_row(GgmlType::IQ2_XXS, row, &mut w).unwrap();
            let (mut exact, mut abs) = (0.0f64, 0.0f64);
            for (b, wb) in w.as_chunks::<256>().0.iter().enumerate() {
                let blk = &acol[296 * b..296 * (b + 1)];
                let dy = f64::from(f32::from_le_bytes(blk[0..4].try_into().unwrap()));
                let wblk = &row[66 * b..66 * (b + 1)];
                let dw = f64::from(half_to_f32(u16::from_le_bytes([wblk[0], wblk[1]])));
                for (n, (&wv, &q)) in wb.iter().zip(&blk[8..264]).enumerate() {
                    // Sub-block n / 32's scale nibble, the top of its word's last byte.
                    let sc = f64::from(2 * (wblk[9 + 8 * (n / 32)] >> 4) + 1);
                    let yq = dy * f64::from(q as i8);
                    let term = f64::from(wv) * yq;
                    exact += term;
                    abs += (f64::from(wv).abs() + 0.125 * dw.abs() * sc * 86.0) * yq.abs();
                }
            }
            let bound = n_u / (1.0 - n_u) * abs;
            for (path, got) in [
                ("dot_row", dot_row(GgmlType::IQ2_XXS, row, &acol, k)),
                (
                    "dot_row_scalar",
                    dot_row_scalar(GgmlType::IQ2_XXS, row, &acol, k),
                ),
            ] {
                let got = f64::from(got.unwrap());
                assert!(
                    (got - exact).abs() <= bound,
                    "{path} k = {k} row {r}: got {got:e}, exact {exact:e}, bound {bound:e}"
                );
            }
        }
    }
}

// ---------------------------------------------------------- IQ1_S x Q8_K
// The kernel is ik's `mul_mat_iq1_s_q8_K` and no harness dumps it, so the gates stand on the
// crate's own oracles, as IQ2_XXS's do: kernel vs mirror (bit identity), ours vs the dequant
// oracle (a band, below) and tile vs dot_row (bit identity).

/// An IQ1_S block: f16 `d`, the 32 grid index bytes `idx(4s + l)` (sub-block `s`, group
/// `l`), then the eight `qh` words `qh(s)`: bits `3l .. 3l + 3` group `l`'s high index bits,
/// bits 12..15 the scale, bit 15 the delta's sign.
fn iq1s_block(d: u16, idx: impl Fn(usize) -> u8, qh: impl Fn(usize) -> u16) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    for s in 0..8 {
        b.extend((0..4).map(|l| idx(4 * s + l)));
    }
    for s in 0..8 {
        b.extend_from_slice(&qh(s).to_le_bytes());
    }
    b
}

/// The band gate's hand-built block: sub-block `s` reads group `l` at index
/// `((s + l) & 7) << 8 | idx(4s + l)`, scale `s`, and a negative delta on odd sub-blocks.
fn iq1s_hand_block() -> Vec<u8> {
    iq1s_block(
        F16_ONE,
        |i| ((i * 37 + 11) & 0xff) as u8,
        |s| {
            let high: u16 = (0..4).map(|l| (((s + l) & 7) as u16) << (3 * l)).sum();
            ((s as u16 & 7) << 12) | high | if s % 2 == 1 { 0x8000 } else { 0 }
        },
    )
}

/// IQ1_S end blocks: every index byte and index bit with the top scale under either delta and
/// either sign of `d`, the index bits alone (scale 0), the scale alone over zero indexes, a
/// scrambled pattern, d = 0 and a subnormal d.
const IQ1S_ENDS: [IqEnd; 8] = [
    || iq1s_block(F16_ONE, |_| 0xFF, |_| 0x7FFF),
    || iq1s_block(F16_ONE, |_| 0xFF, |_| 0xFFFF),
    || iq1s_block(F16_NEG_ONE, |_| 0xFF, |_| 0xFFFF),
    || iq1s_block(F16_ONE, |_| 0xFF, |_| 0x0FFF),
    || iq1s_block(F16_ONE, |_| 0, |_| 0x8007),
    || {
        iq1s_block(
            F16_ONE,
            |i| (i * 41 + 7) as u8,
            |s| 0xC3A5u16.rotate_left(3 * s as u32),
        )
    },
    || iq1s_block(0, |_| 0xFF, |_| 0xFFFF),
    || iq1s_block(F16_TINY, |i| i as u8, |_| 0x50F0),
];

/// IQ1_S's k contract: a non-multiple of 256 is `UnalignedK` by name, before any byte is
/// read.
#[test]
fn iq1s_refuses_unaligned_k() {
    let wrow = vec![0u8; 50 * 24];
    let acol = vec![0u8; 296 * 24];
    assert_eq!(
        dot_row(GgmlType::IQ1_S, &wrow, &acol, 6144 + 128),
        Err(QdotError::UnalignedK {
            k: 6144 + 128,
            gran: 256
        })
    );
}

/// IQ1_S kernel vs mirror on [`generated_gate_a`]'s rows (random bytes, finite scales) at
/// 24, three and one blocks, and on the ends of [`IQ1S_ENDS`] (every grid index, index bit,
/// scale and delta sign at its extreme, d = 0, a subnormal d) at one and 24 blocks against
/// the ±3.0 and seeded columns — bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq1s_kernel_matches_mirror() {
    generated_gate_a(GgmlType::IQ1_S, 50, 256, &[6144, 768, 256]);
    for k in [256, 6144] {
        let nb = k / 256;
        let rows = legacy_end_rows(&IQ1S_ENDS, nb, |end| end());
        let cols = legacy_tile_columns(GgmlType::IQ1_S, k, vec![]);
        for (r, row) in rows.chunks(50 * nb).enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let a = dot_row_avx2(GgmlType::IQ1_S, row, col, k).unwrap();
                let b = dot_row_scalar(GgmlType::IQ1_S, row, col, k).unwrap();
                assert!(
                    a.to_bits() == b.to_bits(),
                    "end row {r} (k = {k}) column {j}: kernel {a:e} (bits {:#x}) and mirror {b:e} \
                     (bits {:#x}) must be bit-identical",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        eprintln!(
            "IQ1_S: {} end rows x {} columns bit-identical (k = {k})",
            rows.len() / (50 * nb),
            cols.len()
        );
    }
}

/// IQ1_S against the dequant oracle: `dequant_row`'s f32 weights (held to ggml's
/// `dequantize_row_iq1_s` bit for bit in gguf) times the column's scaled codes, summed
/// exactly in f64, at the hand-built block ([`iq1s_hand_block`]), the ends' one- and
/// three-block rows and four random rows, through `dot_row` and `dot_row_scalar`.
///
/// The oracle's weights carry two f32 roundings a term (`d`, then the value). Ours sums one
/// exact integer a block and lane — the codes `8·(2s + 1)·(grid + 1)` with the fold
/// `(2s + 1)·(δ − 8)·Σ q` of each sub-block, both inside the i32 sum — in one fmadd against
/// the rounded `d·dy`, then the closing hsum's three adds and the exact `0.125`. Those
/// roundings act on sums bounded by `Σ (|w| + 0.125·|d|·(2s + 1)·25)·|dy·q|` over the values
/// (the codes' offset `8·|grid + 1| ≤ 16` and the fold's `9`); the band is `gamma_(nb + 10)`
/// of that sum, the IQ2_XXS test's rule ([`iq2xxs_dot_is_within_its_f64_bound`]) at IQ1_S's
/// one fmadd a block and lane. A wrong grid byte, index bit, scale or delta sign moves a
/// whole term, far past it.
#[test]
fn iq1s_dot_is_within_its_f64_bound() {
    let mut rng = Lcg(0x2C_1F7E);
    for k in [256usize, 768, 6144] {
        let nb = k / 256;
        let x: Vec<f32> = (0..k).map(|_| rng.unit() * 30.0).collect();
        let mut acol = vec![0u8; col_bytes(GgmlType::IQ1_S, k)];
        quantize_col(GgmlType::IQ1_S, &x, &mut acol);
        let mut rows = vec![iq1s_hand_block().repeat(nb)];
        rows.extend(
            legacy_end_rows(&IQ1S_ENDS, nb, |end| end())
                .chunks(50 * nb)
                .map(<[u8]>::to_vec),
        );
        for _ in 0..4 {
            rows.push(iq_random_block(&mut rng, 50, &[0]).repeat(nb));
        }
        let n_u = (nb + 10) as f64 * 2f64.powi(-24);
        for (r, row) in rows.iter().enumerate() {
            let mut w = vec![0.0f32; k];
            dequant_row(GgmlType::IQ1_S, row, &mut w).unwrap();
            let (mut exact, mut abs) = (0.0f64, 0.0f64);
            for (b, wb) in w.as_chunks::<256>().0.iter().enumerate() {
                let blk = &acol[296 * b..296 * (b + 1)];
                let dy = f64::from(f32::from_le_bytes(blk[0..4].try_into().unwrap()));
                let wblk = &row[50 * b..50 * (b + 1)];
                let dw = f64::from(half_to_f32(u16::from_le_bytes([wblk[0], wblk[1]])));
                for (n, (&wv, &q)) in wb.iter().zip(&blk[8..264]).enumerate() {
                    // Sub-block n / 32's scale nibble: bits 12..15 of its qh word.
                    let qh = u16::from_le_bytes([wblk[34 + 2 * (n / 32)], wblk[35 + 2 * (n / 32)]]);
                    let sc = f64::from(2 * ((qh >> 12) & 7) + 1);
                    let yq = dy * f64::from(q as i8);
                    let term = f64::from(wv) * yq;
                    exact += term;
                    abs += (f64::from(wv).abs() + 0.125 * dw.abs() * sc * 25.0) * yq.abs();
                }
            }
            let bound = n_u / (1.0 - n_u) * abs;
            for (path, got) in [
                ("dot_row", dot_row(GgmlType::IQ1_S, row, &acol, k)),
                (
                    "dot_row_scalar",
                    dot_row_scalar(GgmlType::IQ1_S, row, &acol, k),
                ),
            ] {
                let got = f64::from(got.unwrap());
                assert!(
                    (got - exact).abs() <= bound,
                    "{path} k = {k} row {r}: got {got:e}, exact {exact:e}, bound {bound:e}"
                );
            }
        }
    }
}

/// IQ1_S tile clause: random rows (every grid index, index bit, scale and delta sign, `d` of
/// either sign) and the ends of [`IQ1S_ENDS`] at k = 6144 (Mistral Large 4's host experts),
/// 768 and 256 — tile = `dot_row` per column, bit for bit. No IQ1_S file is on the box, so
/// there are no real rows.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq1s_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ1_S, 50, 256, &[6144, 768, 256], &IQ1S_ENDS);
}

// ------------------------------------------------------- Q8_0 x Q8_2_X4
// The oracle is a synthetic block set (tools/ref/q8f0_ref.cpp): its generator writes the
// column's f32 values, ik's q8_2_x4 bytes of it and every weight row into the dump, so this
// side reads the inputs instead of regenerating them. k = 2144 is 16 whole x4 groups and three
// tail blocks. The set carries -128 weight codes, zero, subnormal and negative f16 scales, a
// zero activation block, and block 9, whose bf16 scale is the subnormal 2^-127 and whose first
// code is -128: under a negative weight code that product takes the sign trick's wrap, in ik's
// kernel and in ours alike. At d = 2^-127 the wrap's term sits below every row sum's f32
// resolution, so block 9 cannot tell a wrap from a saturation; gate A pins the wrap on crafted
// columns at a unit scale instead. Gate B is bit identity: ours is ik's instruction graph, with
// the same operand order at every float operation, so no rounding can differ.

/// The synthetic Q8_0 dump: `k`, the column's f32 values, ik's q8_2_x4 bytes of the column,
/// the weight rows and ik's result bits per row.
struct Q8f0Dump {
    k: usize,
    x: Vec<f32>,
    ik_acol: Vec<u8>,
    rows: Vec<Vec<u8>>,
    want: Vec<u32>,
}

/// Lower-case hex to bytes.
fn unhex(h: &str) -> Vec<u8> {
    assert!(h.len().is_multiple_of(2), "odd hex length {}", h.len());
    h.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).expect("hex digits"))
        .collect()
}

/// Reads `q8f0-ik-dot.txt`, every line by its tag; an untagged or unknown line is refused.
fn q8f0_dump() -> Q8f0Dump {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let path = format!("{base}/ref/q8f0-ik-dot.txt");
    let dump = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e} — run just build-ref first"));
    let mut d = Q8f0Dump {
        k: 0,
        x: Vec::new(),
        ik_acol: Vec::new(),
        rows: Vec::new(),
        want: Vec::new(),
    };
    for line in dump.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["tensor", "synthetic-q8_0", "k", k] => d.k = k.parse().unwrap(),
            ["x", h] => {
                d.x = unhex(h)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
            }
            ["a", h] => d.ik_acol = unhex(h),
            ["w", r, h] => {
                assert_eq!(r.parse::<usize>().unwrap(), d.rows.len(), "rows in order");
                d.rows.push(unhex(h));
            }
            ["row", r, bits] => {
                assert_eq!(
                    r.parse::<usize>().unwrap(),
                    d.want.len(),
                    "results in order"
                );
                d.want
                    .push(u32::from_str_radix(bits, 16).expect("ik dumps raw f32 bits"));
            }
            _ => panic!("{path}: an unknown line {:?}", &line[..line.len().min(40)]),
        }
    }
    assert_eq!(d.k, 2144, "{path}: the harness's k");
    assert_eq!(d.x.len(), d.k, "{path}: the column");
    assert_eq!(
        d.ik_acol.len(),
        col_bytes(GgmlType::Q8_0, d.k),
        "{path}: ik's column"
    );
    assert_eq!(d.rows.len(), 64, "{path}: rows");
    assert_eq!(d.want.len(), 64, "{path}: results");
    for (r, w) in d.rows.iter().enumerate() {
        assert_eq!(w.len(), 34 * d.k / 32, "{path}: row {r}");
    }
    d
}

/// Q8_0 gate 0: `quantize_col` codes the synthetic column byte for byte as ik's
/// `quantize_row_q8_2_x4`, and ik's bytes carry the edge the set was built for — block 9's
/// bf16 scale the subnormal 2^-127 (bits 0x0040) and its first code -128.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_q8f0_encoder_matches_ik() {
    let d = q8f0_dump();
    let mut acol = vec![0u8; col_bytes(GgmlType::Q8_0, d.k)];
    quantize_col(GgmlType::Q8_0, &d.x, &mut acol);
    for (i, (a, b)) in d.ik_acol.iter().zip(&acol).enumerate() {
        assert_eq!(a, b, "encoder byte {i}: ik {a:02x} vs ours {b:02x}");
    }
    // Block 9 is block 1 of group 2: its d at group byte 2, its codes from byte 48.
    let g = &d.ik_acol[2 * 144..3 * 144];
    assert_eq!(
        u16::from_le_bytes([g[2], g[3]]),
        0x0040,
        "block 9's scale is the subnormal bf16 2^-127"
    );
    assert_eq!(g[48], 0x80, "block 9's first code is -128");
    eprintln!(
        "Q8_0 gate 0: {} encoder bytes bit-identical to ik's quantize_row_q8_2_x4 \
         (k = {}, three tail blocks, block 9 at d = 2^-127 with a -128 code)",
        acol.len(),
        d.k
    );
}

/// Q8_0 gate A: the AVX2 kernel and its mirror agree bit for bit — on the dump's 64 rows
/// against ik's column, and on `ROWS` generated rows at every tail count (k = 32, 96, 128,
/// 160, 224 and 2144: 1, 3, 0, 1, 3 and 3 tail blocks) against columns of every magnitude.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_q8f0_kernel_matches_mirror() {
    assert!(
        supports(GgmlType::Q8_0),
        "gate A compares the AVX2 kernel against its mirror; this CPU lacks the kernel's ISA"
    );
    let same = |w: &[u8], a: &[u8], k: usize, what: &str| {
        let p = dot_row_avx2(GgmlType::Q8_0, w, a, k).unwrap();
        let q = dot_row_scalar(GgmlType::Q8_0, w, a, k).unwrap();
        assert!(
            p.to_bits() == q.to_bits(),
            "{what}: kernel {p:e} (bits {:#x}) and mirror {q:e} (bits {:#x}) must be bit-identical",
            p.to_bits(),
            q.to_bits()
        );
    };
    let d = q8f0_dump();
    for (r, w) in d.rows.iter().enumerate() {
        same(w, &d.ik_acol, d.k, &format!("dump row {r}"));
    }
    let mut rng = Lcg(0x0008_F0DA_7A5E);
    let mut n = 0;
    for k in [32usize, 96, 128, 160, 224, 2144] {
        for c in 0..ROWS / 16 {
            let mag = [1e-3f32, 1.0, 1e3][c % 3];
            let x: Vec<f32> = (0..k).map(|_| rng.unit() * mag).collect();
            let mut a = vec![0u8; col_bytes(GgmlType::Q8_0, k)];
            quantize_col(GgmlType::Q8_0, &x, &mut a);
            let mut w: Vec<u8> = (0..34 * k / 32).map(|_| rng.next_u32() as u8).collect();
            // Finite f16 scales: a random exponent below 0x1f, sign and mantissa kept.
            for b in w.as_chunks_mut::<34>().0 {
                let h = u16::from_le_bytes([b[0], b[1]]);
                let h = (h & 0x83ff) | ((h % 0x1f) << 10);
                b[0..2].copy_from_slice(&h.to_le_bytes());
            }
            same(&w, &a, k, &format!("generated k = {k} column {c}"));
            n += 1;
        }
    }
    // The encoder emits -128 only where its scale underflows, so no encoded column shows the
    // sign trick's wrap in an f32 sum. Crafted columns pin it: activation code -128 under a unit
    // bf16 scale, met by weight code -127 under a unit f16 scale, in an x4 group and in a tail
    // block. There the wrap (+128 -> -128) and a saturation (127) differ by 255 · 127.
    let mut wraps = 0;
    for (k, blk) in [(32usize, 0usize), (160, 0), (160, 4), (2144, 9)] {
        let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
        let mut a = vec![0u8; col_bytes(GgmlType::Q8_0, k)];
        quantize_col(GgmlType::Q8_0, &x, &mut a);
        let nb4 = k / 32 / 4 * 4;
        let (d_at, q_at) = if blk < nb4 {
            let g = blk / 4 * 144;
            (g + 2 * (blk % 4), g + 16 + 32 * (blk % 4))
        } else {
            let t = nb4 / 4 * 144 + (blk - nb4) * 36;
            (t, t + 4)
        };
        a[d_at..d_at + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
        a[q_at] = 0x80;
        let mut w: Vec<u8> = (0..34 * k / 32).map(|_| rng.next_u32() as u8).collect();
        for b in w.as_chunks_mut::<34>().0 {
            let h = u16::from_le_bytes([b[0], b[1]]);
            let h = (h & 0x83ff) | ((h % 0x1f) << 10);
            b[0..2].copy_from_slice(&h.to_le_bytes());
        }
        w[34 * blk..34 * blk + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        w[34 * blk + 2] = 0x81;
        let what = format!("crafted wrap k = {k} block {blk}");
        same(&w, &a, k, &what);
        // The site shows in the sum: -127 in the code's place moves the result.
        let mut b = a.clone();
        b[q_at] = 0x81;
        let wrapped = dot_row_scalar(GgmlType::Q8_0, &w, &a, k).unwrap();
        let unwrapped = dot_row_scalar(GgmlType::Q8_0, &w, &b, k).unwrap();
        assert!(
            wrapped.to_bits() != unwrapped.to_bits(),
            "{what}: codes -128 and -127 give the same sum {wrapped:e}; the site is invisible"
        );
        wraps += 1;
    }
    eprintln!(
        "Q8_0 gate A: 64 dump rows, {n} generated rows and {wraps} crafted -128 wrap rows \
         bit-identical (kernel vs mirror, tails of 0 to 3 blocks)"
    );
}

/// Q8_0 gate B: on ik's column, every dumped row bit-identical to ik's
/// `mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1, block_q8_2>`.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_q8f0_kernel_predicts_ik() {
    let d = q8f0_dump();
    // Rows whose block-9 first weight code is negative under a nonzero scale: the -128
    // activation code meets the sign trick's wrap there.
    let wraps = d
        .rows
        .iter()
        .filter(|w| {
            let b = &w[9 * 34..10 * 34];
            (b[2] as i8) < 0 && u16::from_le_bytes([b[0], b[1]]) & 0x7fff != 0
        })
        .count();
    assert!(wraps > 0, "the set must carry the sign trick's wrap");
    for (r, (w, &want)) in d.rows.iter().zip(&d.want).enumerate() {
        let got = dot_row(GgmlType::Q8_0, w, &d.ik_acol, d.k).unwrap();
        assert!(
            got.to_bits() == want,
            "row {r}: ours {got:.9e} (bits {:#010x}) vs ik {:.9e} (bits {want:#010x})",
            got.to_bits(),
            f32::from_bits(want)
        );
    }
    eprintln!(
        "Q8_0 gate B: 64 rows bit-identical to ik's mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1, \
         block_q8_2> (on ik's own column, {wraps} rows meet the -128 code at d = 2^-127)"
    );
}

// ------------------------------------------------------------- Q2_0 x Q8_0
// Mainline ggml's Q2_0 (llama.cpp f2cc7282c; ik has no Q2_0). Each constant below is mainline's
// own output over the inputs its clause builds, printed on x86_64 Linux by a short C program
// linked against that build's libggml-cpu.so (GGML_NATIVE on, so its x86 paths): the Q8_0
// traits' `from_float`, x86 `quantize_row_q8_0` (ggml-cpu/arch/x86/quants.c:302), and the Q2_0
// traits' `vec_dot`, `ggml_vec_dot_q2_0_q8_0`. These clauses need no box: `quantize_col` and
// `dot_row` take the AVX2 paths on a CPU with them, and the scalar mirrors are held to the same
// bytes and bits.

/// One LCG draw as a byte: [`Lcg::next_u32`] keeps the state's top 31 bits, so their top eight.
fn q2f0_byte(rng: &mut Lcg) -> u8 {
    (rng.next_u32() >> 23) as u8
}

/// One LCG draw as an f16 scale of either sign and magnitude in [2^-10, 2^3): the sign from bit
/// 0, the exponent field `5 + (r >> 1) % 13`, the mantissa from bits 20..30.
fn q2f0_scale(rng: &mut Lcg) -> u16 {
    let r = rng.next_u32();
    (((r & 1) << 15) | ((5 + (r >> 1) % 13) << 10) | ((r >> 20) & 0x3ff)) as u16
}

/// [`q2_0_column_matches_mainline`]'s 256 values, eight q8_0 blocks: LCG values in [-1, 1)
/// times 1, 100, 1e-3, 1, 1e4, 0, 1e-9 and 30, block by block; then block 0 is -127 and the
/// ties 0.5 ..= 30.5 (at amax 127 the multiplier is 1), and block 1 holds its amax, 100, at
/// value 5.
fn q2f0_column_values() -> Vec<f32> {
    let mags = [1.0f32, 100.0, 1e-3, 1.0, 1e4, 0.0, 1e-9, 30.0];
    let mut rng = Lcg(0x0800_5EED);
    let mut x: Vec<f32> = (0..256)
        .map(|i| (rng.next_u32() as f32 / 1073741824.0 - 1.0) * mags[i / 32])
        .collect();
    x[0] = -127.0;
    for (j, v) in x[1..32].iter_mut().enumerate() {
        *v = j as f32 + 0.5;
    }
    x[37] = 100.0;
    x
}

/// Mainline's x86 `quantize_row_q8_0` over [`q2f0_column_values`]: eight 34-byte blocks, two
/// lines a block.
#[rustfmt::skip]
const Q8F0_MAINLINE: [u8; 272] = [
        0x00, 0x3c, 0x81, 0x00, 0x02, 0x02, 0x04, 0x04, 0x06, 0x06, 0x08, 0x08, 0x0a, 0x0a, 0x0c, 0x0c, 0x0e,
        0x0e, 0x10, 0x10, 0x12, 0x12, 0x14, 0x14, 0x16, 0x16, 0x18, 0x18, 0x1a, 0x1a, 0x1c, 0x1c, 0x1e, 0x1e,
        0x4d, 0x3a, 0x01, 0xf5, 0x7a, 0x61, 0xb7, 0x7f, 0xac, 0x49, 0xd8, 0xc8, 0xbe, 0x0c, 0xf1, 0x9b, 0x6e,
        0x46, 0x34, 0x54, 0x14, 0x2b, 0x0b, 0xf5, 0x4c, 0xfd, 0x2b, 0x60, 0x0b, 0x59, 0x6d, 0xda, 0xf7, 0x12,
        0x81, 0x00, 0x0d, 0x5a, 0xa3, 0x1f, 0x2c, 0xf6, 0xc1, 0xb5, 0x3c, 0x0b, 0xfb, 0x7b, 0xb1, 0x48, 0xf1,
        0x05, 0x58, 0xa1, 0x9e, 0x6a, 0x4e, 0xeb, 0xfd, 0xc0, 0xd9, 0x6e, 0x49, 0x81, 0x13, 0x50, 0x3c, 0xe3,
        0xe4, 0x1f, 0xe0, 0x7d, 0xd3, 0xf1, 0xba, 0xe8, 0xd8, 0xaf, 0xeb, 0xaa, 0xb9, 0x8f, 0x9b, 0x7e, 0xfa,
        0x4f, 0xcb, 0xe8, 0x7f, 0x9e, 0x06, 0x97, 0x20, 0xee, 0x3a, 0x1a, 0x34, 0xc1, 0xa9, 0x08, 0x07, 0xd4,
        0xdd, 0x54, 0xd6, 0xd5, 0xa5, 0x1f, 0xbb, 0x05, 0x54, 0x09, 0x0f, 0xf4, 0x14, 0xbb, 0x37, 0x76, 0xcf,
        0x5b, 0xc7, 0x06, 0xf6, 0x8b, 0x27, 0x81, 0xe8, 0x15, 0x65, 0xd1, 0x79, 0xa6, 0xad, 0x4d, 0x52, 0x75,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x1d, 0x92, 0x83, 0x93, 0x11, 0x17, 0x97, 0x7f, 0xeb, 0x55, 0x63, 0x10, 0xaf, 0x3d, 0x2e,
        0xc0, 0xe4, 0x19, 0xe9, 0x25, 0xec, 0xd3, 0x21, 0x1f, 0xc1, 0xb9, 0x3f, 0xf7, 0x44, 0xcc, 0x2d, 0x7a,
        0x66, 0x33, 0xa3, 0x57, 0x09, 0x8a, 0x58, 0xfd, 0x3d, 0x7f, 0xbf, 0x47, 0x44, 0xf8, 0xfe, 0x82, 0x7a,
        0xdf, 0x44, 0x35, 0xe6, 0x1c, 0xda, 0x81, 0x31, 0x5f, 0x87, 0x36, 0xa9, 0x8c, 0x96, 0x09, 0x59, 0xd6,
];

/// The Q2_0 activation column against mainline's x86 encoder: `quantize_col` and its scalar
/// mirror write [`Q8F0_MAINLINE`] byte for byte. That pins the f16 scale, the multiplier
/// `127 / amax` (block 6's scale rounds to an f16 zero beside nonzero codes; coding with
/// `1 / d` of the stored f16 scale changes 34 bytes) and ties to even (block 0; `roundf`
/// changes 16, and so does `quantize_row_q8_0_ref`, `1 / d` of the f32 scale with `roundf`).
/// The codes restore to `quantize_row_q8_0_roundtrip`'s values, equal as values (a code holds
/// no zero sign).
#[test]
fn q2_0_column_matches_mainline() {
    type Encode = fn(GgmlType, &[f32], &mut [u8]);
    let x = q2f0_column_values();
    let mut col = vec![0u8; col_bytes(GgmlType::Q2_0, 256)];
    for (path, encode) in [
        ("quantize_col", quantize_col as Encode),
        ("quantize_col_scalar", quantize_col_scalar as Encode),
    ] {
        encode(GgmlType::Q2_0, &x, &mut col);
        for (i, (&got, &want)) in col.iter().zip(&Q8F0_MAINLINE).enumerate() {
            assert!(
                got == want,
                "{path}: block {} byte {}: got {got:#04x}, want {want:#04x}",
                i / 34,
                i % 34
            );
        }
    }
    let mut rt = vec![0.0f32; 256];
    quantize_row_q8_0_roundtrip(&x, &mut rt);
    for (b, blk) in col.as_chunks::<34>().0.iter().enumerate() {
        let scale_bits = u16::from_le_bytes([blk[0], blk[1]]);
        assert_eq!(q8f0_block(|i| blk[2 + i] as i8, scale_bits), *blk);
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for (j, &q) in blk[2..].iter().enumerate() {
            let v = d * f32::from(q as i8);
            let p = 32 * b + j;
            assert!(
                v == rt[p],
                "value {p}: restored {v:e}, round trip {:e}",
                rt[p]
            );
        }
    }
}

/// Width of [`q2_0_dot_matches_mainline`]'s rows: 32 Q2_0 blocks against 64 q8_0 blocks.
const Q2F0_K: usize = 2048;

/// [`q2_0_dot_matches_mainline`]'s q8_0 column and 16 Q2_0 rows, from one LCG: per q8_0 block
/// a [`q2f0_scale`] then 32 code bytes (every i8, -128 included), per Q2_0 block a scale then
/// 16 code bytes. Then the ends: column block 7's scale is 0 and block 21's the subnormal
/// 0x0001; row 12's codes are all 0 (every value -1), row 13's all 3 (+2), row 14's scales
/// alternate 0 and 0x0001, row 15's +65504 and -65504.
fn q2f0_dot_case() -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut rng = Lcg(0x0200_5EED);
    let mut col = Vec::with_capacity(Q2F0_K / 32 * 34);
    for b in 0..Q2F0_K / 32 {
        let d = match (b, q2f0_scale(&mut rng)) {
            (7, _) => 0,
            (21, _) => 1,
            (_, d) => d,
        };
        col.extend_from_slice(&d.to_le_bytes());
        for _ in 0..32 {
            col.push(q2f0_byte(&mut rng));
        }
    }
    let mut rows = Vec::new();
    for r in 0..16 {
        let mut row = Vec::with_capacity(Q2F0_K / 64 * 18);
        for b in 0..Q2F0_K / 64 {
            let d = match (r, q2f0_scale(&mut rng)) {
                (14, _) => [0x0000, 0x0001][b % 2],
                (15, _) => [0x7bff, 0xfbff][b % 2],
                (_, d) => d,
            };
            row.extend_from_slice(&d.to_le_bytes());
            for _ in 0..16 {
                row.push(match (r, q2f0_byte(&mut rng)) {
                    (12, _) => 0x00,
                    (13, _) => 0xff,
                    (_, c) => c,
                });
            }
        }
        rows.push(row);
    }
    (col, rows)
}

/// Mainline's `ggml_vec_dot_q2_0_q8_0` over [`q2f0_dot_case`]'s rows, as f32 bits.
#[rustfmt::skip]
const Q2F0_DOT_MAINLINE: [u32; 16] = [
        0x4506affc, 0x45cc669b, 0xc6a04ab9, 0x473a069d,
        0x4701ebbd, 0xc521ce36, 0x45b64c0c, 0x46385bb2,
        0x4621ff41, 0xc58518d8, 0x4652919b, 0x44957542,
        0xc43e724d, 0x464c96ee, 0x3a5af47c, 0x4cd2fdf0,
];

/// Q2_0 x Q8_0 against mainline's dot, bit for bit, through `dot_row` (the AVX2 kernel on a CPU
/// with it) and its scalar mirror. Mainline's x86 build runs the generic loop
/// (ggml-cpu/quants.c:177) with its three float steps fused, and these rows tell that order
/// apart: the same loop with plain steps differs on 10 of the 16, with eight blocks summed in
/// lanes on 14.
#[test]
fn q2_0_dot_matches_mainline() {
    let (col, rows) = q2f0_dot_case();
    for (r, (row, &want)) in rows.iter().zip(&Q2F0_DOT_MAINLINE).enumerate() {
        for (path, got) in [
            ("dot_row", dot_row(GgmlType::Q2_0, row, &col, Q2F0_K)),
            (
                "dot_row_scalar",
                dot_row_scalar(GgmlType::Q2_0, row, &col, Q2F0_K),
            ),
        ] {
            let got = got.unwrap();
            assert!(
                got.to_bits() == want,
                "{path} row {r}: got {got:e} (bits {:#010x}), want {:e} (bits {want:#010x})",
                got.to_bits(),
                f32::from_bits(want)
            );
        }
    }
}

/// A q8_0 column as the dot reads it: per 34-byte block the f16 scale times each i8 code, in
/// f64 (a product of an 11-bit scale and an 8-bit code is exact).
fn restore_q8f0(acol: &[u8]) -> Vec<f64> {
    let mut out = Vec::with_capacity(acol.len() / 34 * 32);
    for blk in acol.as_chunks::<34>().0 {
        let d = f64::from(half_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
        out.extend(blk[2..].iter().map(|&q| d * f64::from(q as i8)));
    }
    out
}

/// Q2_0 x Q8_0 against an exact f64 dot, the check that stands on neither mainline's float
/// order nor the mirror: `dequant_row` (held to mainline's `to_float` in gguf) times
/// [`restore_q8f0`]'s column, at the widths no constant above covers (one, three and 32 Q2_0
/// blocks), through `dot_row` and `dot_row_scalar`. The terms are one product sum per q8_0
/// block, exact in f64 (each product has at most 32 significant bits, and a block's sum is an
/// integer multiple of one scale pair). The f32 dot rounds a term in its `sumi` step (twice for
/// a block's first q8_0 half), in its Q2_0 block's `sumf` step and in each later block's: at
/// most `nb + 2` roundings of relative 2^-24, so the bound is `gamma_(nb+2) * sum |term|` with
/// `gamma_n = n u / (1 - n u)`. The f64 sum's own error is below 1e-14 of the same sum, far
/// inside it. A wrong code map or bit order is off by whole terms.
#[test]
fn q2_0_dot_is_within_its_f64_bound() {
    let (col, rows) = q2f0_dot_case();
    for k in [64, 192, Q2F0_K] {
        let nb = k / 64;
        let acol = &col[..col_bytes(GgmlType::Q2_0, k)];
        let y = restore_q8f0(acol);
        let n_u = (nb + 2) as f64 * 2f64.powi(-24);
        for (r, row) in rows.iter().enumerate() {
            let row = &row[..18 * nb];
            let mut w = vec![0.0f32; k];
            dequant_row(GgmlType::Q2_0, row, &mut w).unwrap();
            let (mut exact, mut abs) = (0.0f64, 0.0f64);
            for (wb, yb) in w.as_chunks::<32>().0.iter().zip(y.as_chunks::<32>().0) {
                let term: f64 = wb.iter().zip(yb).map(|(&a, &b)| f64::from(a) * b).sum();
                exact += term;
                abs += term.abs();
            }
            let bound = n_u / (1.0 - n_u) * abs;
            for (path, got) in [
                ("dot_row", dot_row(GgmlType::Q2_0, row, acol, k)),
                (
                    "dot_row_scalar",
                    dot_row_scalar(GgmlType::Q2_0, row, acol, k),
                ),
            ] {
                let got = f64::from(got.unwrap());
                assert!(
                    (got - exact).abs() <= bound,
                    "{path} k = {k} row {r}: got {got:e}, exact {exact:e}, bound {bound:e}"
                );
            }
        }
    }
}

/// Q2_0 kernel vs mirror on [`generated_gate_a`]'s rows (random codes, finite scales) at 32,
/// one and three blocks against a `quantize_col` column, bit for bit.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_q2_0_kernel_matches_mirror() {
    generated_gate_a(GgmlType::Q2_0, 18, 64, &[2048, 64, 192]);
}

// ------------------------------------------------- q_nope2 cell kernel
// Q8_0 x act cell kernel gates: AVX2 kernel vs scalar mirror, bit for bit,
// over model and boundary shapes and extreme code patterns.

/// Deterministic xorshift64* generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform code in [-127, 127].
    #[cfg(target_arch = "x86_64")]
    fn code(&mut self) -> i8 {
        ((self.next() % 255) as i32 - 127) as i8
    }
}

/// Weight f16 scales and activation f32 scales.
#[cfg(target_arch = "x86_64")]
const W_SCALES: [u16; 6] = [0x3C00, 0x3800, 0x4000, 0x3B00, 0x0040, 0xBC00];
#[cfg(target_arch = "x86_64")]
const A_SCALES: [f32; 6] = [1.0, 0.5, 2.0, 0.00390625, 0.25, -1.0];

#[cfg(target_arch = "x86_64")]
fn gen_blocks(rng: &mut Rng, n: usize, side: u32) -> Vec<qdot::Q8Block> {
    (0..n)
        .map(|i| {
            let mut q = [0i8; 32];
            for c in &mut q {
                *c = rng.code();
            }
            qdot::Q8Block {
                d: W_SCALES[(i % W_SCALES.len() + side as usize) % W_SCALES.len()],
                q,
            }
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn gen_acol(rng: &mut Rng, nb: usize) -> Vec<qdot::ActBlock> {
    (0..nb)
        .map(|i| {
            let mut q = [0i8; 32];
            for c in &mut q {
                *c = rng.code();
            }
            qdot::ActBlock {
                d: A_SCALES[i % A_SCALES.len()],
                q,
            }
        })
        .collect()
}

/// Compare kernel vs mirror over one (whead, acol) pair on bits.
#[cfg(target_arch = "x86_64")]
fn assert_cells_bit_identical(
    what: &str,
    whead: &[qdot::Q8Block],
    acol: &[qdot::ActBlock],
    segments: &[(usize, usize)],
) {
    let latent = whead.len() / acol.len();
    for &(j0, j_end) in segments {
        let len = j_end - j0;
        let mut k = vec![0.0f32; len];
        let mut s = vec![0.0f32; len];
        qdot::q_nope2_cells_avx2(whead, acol, j0, j_end, &mut k);
        qdot::q_nope2_cells_scalar(whead, acol, j0, j_end, &mut s);
        for (jj, (a, b)) in k.iter().zip(&s).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: cell {j0}+{jj} (latent {latent}, nb {}): kernel and mirror must be bit-identical",
                acol.len()
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box's AVX2 (q_nope2_cells_avx2 panics without it)"]
fn hw_q_nope2_cells_bit_identical() {
    // Model shape: latent 512, nb = nope/32 = 4.
    let mut rng = Rng(0x5EED_3838_1234_5678);
    let (latent, nb) = (512usize, 4usize);
    let whead = gen_blocks(&mut rng, latent * nb, 0);
    let acol = gen_acol(&mut rng, nb);
    assert_cells_bit_identical(
        "model shape, random codes",
        &whead,
        &acol,
        &[(0, latent), (1, latent - 1), (137, 349), (511, 512), (0, 1)],
    );

    // Extreme patterns over the same shape.
    let consts: &[(&str, i8, i8)] = &[
        ("w+127 a+127 (max positive isum/pairs)", 127, 127),
        ("w-127 a-127 (fold on every lane)", -127, -127),
        ("w-127 a+127 (max negative isum)", -127, 127),
        ("w+127 a-127", 127, -127),
        ("w 0 a+127 (sign fold zeroes a)", 0, 127),
        ("w+127 a 0", 127, 0),
        ("w-128 a+127 (|w|=128 wraps into u8)", -128, 127),
        ("w-128 a-127", -128, -127),
    ];
    for &(what, wc, ac) in consts {
        let whead: Vec<qdot::Q8Block> = (0..latent * nb)
            .map(|i| qdot::Q8Block {
                d: W_SCALES[i % W_SCALES.len()],
                q: [wc; 32],
            })
            .collect();
        let acol: Vec<qdot::ActBlock> = (0..nb)
            .map(|i| qdot::ActBlock {
                d: A_SCALES[i % A_SCALES.len()],
                q: [ac; 32],
            })
            .collect();
        assert_cells_bit_identical(what, &whead, &acol, &[(0, latent), (3, latent)]);
    }

    // Alternating signs for pair cancellation in maddubs.
    let alt = |i: usize| if i.is_multiple_of(2) { 127 } else { -127 };
    let whead: Vec<qdot::Q8Block> = (0..latent * nb)
        .map(|i| qdot::Q8Block {
            d: W_SCALES[i % W_SCALES.len()],
            q: core::array::from_fn(alt),
        })
        .collect();
    let acol: Vec<qdot::ActBlock> = (0..nb)
        .map(|i| qdot::ActBlock {
            d: A_SCALES[i % A_SCALES.len()],
            q: core::array::from_fn(alt),
        })
        .collect();
    assert_cells_bit_identical("alternating ±127", &whead, &acol, &[(0, latent)]);

    // Boundary shapes: nb = 1, odd nb = 3, small latents, single-cell segments.
    for (latent2, nb2) in [(8usize, 1usize), (5, 3), (1, 1), (33, 2)] {
        let mut rng = Rng(0xD00D_0000_0000 + latent2 as u64 * 100 + nb2 as u64);
        let whead = gen_blocks(&mut rng, latent2 * nb2, 1);
        let acol = gen_acol(&mut rng, nb2);
        assert_cells_bit_identical(
            &format!("shape latent={latent2} nb={nb2}"),
            &whead,
            &acol,
            &[(0, latent2), (0, 1), (latent2 - 1, latent2)],
        );
    }

    // Hand-computed absolute value: w = a = +127, d_w = d_a = 1.0 -> 32*127^2 = 516128.0.
    let whead = [qdot::Q8Block {
        d: 0x3C00,
        q: [127; 32],
    }];
    let acol = [qdot::ActBlock {
        d: 1.0,
        q: [127; 32],
    }];
    let mut out = [0.0f32; 1];
    qdot::q_nope2_cells_avx2(&whead, &acol, 0, 1, &mut out);
    assert_eq!(
        out[0].to_bits(),
        516128.0f32.to_bits(),
        "hand value 516128.0"
    );
    qdot::q_nope2_cells_scalar(&whead, &acol, 0, 1, &mut out);
    assert_eq!(out[0].to_bits(), 516128.0f32.to_bits(), "mirror hand value");

    // Dispatcher matches scalar mirror bit for bit.
    let mut rng = Rng(0xABCD_EF01);
    let whead = gen_blocks(&mut rng, latent * nb, 2);
    let acol = gen_acol(&mut rng, nb);
    let mut d = vec![0.0f32; latent];
    let mut s = vec![0.0f32; latent];
    qdot::q_nope2_cells(&whead, &acol, 0, latent, &mut d);
    qdot::q_nope2_cells_scalar(&whead, &acol, 0, latent, &mut s);
    for (a, b) in d.iter().zip(&s) {
        assert_eq!(a.to_bits(), b.to_bits(), "dispatcher vs mirror");
    }

    eprintln!(
        "q_nope2 cells: kernel == scalar mirror, bit for bit, over the model shape, extremes and boundary shapes"
    );
}

/// Dispatcher shape contract: panics on out-of-bounds arguments.
#[test]
fn q_nope2_cells_rejects_bad_shapes() {
    let whead = [qdot::Q8Block {
        d: 0x3C00,
        q: [1; 32],
    }; 8];
    let acol = [qdot::ActBlock { d: 1.0, q: [1; 32] }; 2];
    let mut out = [0.0f32; 4];
    let run = |j0, j_end, o: &mut [f32]| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            qdot::q_nope2_cells(&whead, &acol, j0, j_end, o)
        }))
    };
    run(5, 4, &mut out).expect_err("j0 > j_end must panic");
    run(0, 5, &mut out).expect_err("j_end past whead must panic");
    run(0, 4, &mut out[..3]).expect_err("short out must panic");
    // Legal boundary shape.
    run(0, 4, &mut out).expect("legal shape must not panic");
}

/// The eight-lane SwiGLU against the scalar libm form, and the tail rule: an
/// element's value does not depend on its position in the block.
#[test]
fn swiglu_matches_scalar_and_is_position_independent() {
    let n = 1408 + 5;
    let gate: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 2001) as f32 - 1000.0) * 0.02)
        .collect();
    let up: Vec<f32> = (0..n)
        .map(|i| ((i * 91 % 1777) as f32 - 888.0) * 0.003)
        .collect();
    let mut out = vec![0.0f32; n];
    qdot::swiglu(&gate, &up, &mut out);
    for i in 0..n {
        let want = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
        let tol = 2e-6 * want.abs().max(1e-3);
        assert!(
            (out[i] - want).abs() <= tol,
            "i={i} got {} want {want}",
            out[i]
        );
        let mut one = [0.0f32];
        qdot::swiglu(&gate[i..i + 1], &up[i..i + 1], &mut one);
        assert_eq!(
            one[0].to_bits(),
            out[i].to_bits(),
            "i={i} moves with its position"
        );
    }
    // The saturating ends: exp overflow must give 0 and x, not NaN.
    let mut ends = [0.0f32; 2];
    qdot::swiglu(&[-200.0, 200.0], &[1.0, 1.0], &mut ends);
    assert_eq!(ends, [-0.0, 200.0]);
}

/// The clamped combine reports a NaN or an infinity in `gate` or `up` at every position (each
/// lane of both vectors and the padded tail), which it writes as a finite value, and reports
/// finite inputs as finite. Below the limit nothing is clamped: `true`, the value non-finite in
/// `out`.
#[test]
fn swiglu_clamp_flags_a_non_finite_input_at_every_position() {
    let n = 19;
    let mut out = vec![0.0f32; n];
    let base = vec![f32::MAX; n];
    assert!(qdot::swiglu_clamp(&base, &base, 7.0, &mut out));
    for i in 0..n {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut v = base.clone();
            v[i] = bad;
            assert!(
                !qdot::swiglu_clamp(&v, &base, 7.0, &mut out),
                "gate[{i}] = {bad}"
            );
            assert!(
                !qdot::swiglu_clamp(&base, &v, 7.0, &mut out),
                "up[{i}] = {bad}"
            );
            assert!(
                out[i].is_finite(),
                "up[{i}] = {bad}: the clamp writes a finite value"
            );
            assert!(qdot::swiglu_clamp(&base, &v, 0.0, &mut out) && !out[i].is_finite());
        }
    }
}

// ------------------------------------------------- clamped SwiGLU (V4.1)
// The reference below is ik's rule written out again in scalar f32 from ik's
// source, not a call into qdot: a clamp deleted from `swiglu_clamp` cannot be
// deleted from this transcription too.

/// ik's AVX2 `v_expf` (`ggml/src/iqk/iqk_utils.h:170-222`), one lane: each
/// `_mm256_fmadd_ps`/`_mm256_fnmadd_ps` a `mul_add`, the scale `2^n` built in
/// the exponent bits of `z`, and the two escapes for `|n| > 126` (split scale)
/// and `|n| > 192` (overflow, underflow). The constants are ik's hex floats as
/// bit patterns.
fn ik_expf(x: f32) -> f32 {
    let c = f32::from_bits;
    let r = c(0x4B40_0000); // 0x1.8p23
    let z = x.mul_add(c(0x3FB8_AA3B), r); // 0x1.715476p+0
    let n = z - r;
    // fnmadd(n, 0x1.7f7d1cp-20, fnmadd(n, 0x1.62e4p-1, x))
    let b = (-n).mul_add(c(0x35BF_BE8E), (-n).mul_add(c(0x3F31_7200), x));
    let e = z.to_bits() << 23;
    let k = c(e.wrapping_add(1.0f32.to_bits()));
    let u = b * b;
    // fmadd(fmadd(fmadd(0x1.0e4020p-7, b, 0x1.573e2ep-5), u,
    //             fmadd(0x1.555e66p-3, b, 0x1.fffdb6p-2)), u, 0x1.ffffecp-1 * b)
    let j = c(0x3C07_2010)
        .mul_add(b, c(0x3D2B_9F17))
        .mul_add(u, c(0x3E2A_AF33).mul_add(b, c(0x3EFF_FEDB)))
        .mul_add(u, c(0x3F7F_FFF6) * b);
    // `_CMP_GT_OQ`: false on a NaN `n`, which takes the main path.
    if n.abs() > 126.0 {
        let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
        let s1 = c(g.wrapping_add(0x7F00_0000));
        let s2 = c(e.wrapping_sub(g));
        return if n.abs() > 192.0 {
            s1 * s1
        } else {
            s2.mul_add(j, s2) * s1
        };
    }
    j.mul_add(k, k)
}

/// ik's `mul_mat_up_gate_NxM` for one value (`ggml/src/iqk/iqk_mul_mat.cpp`
/// :153-171): `tmp = silu(gate)` (`v_silu`: `x / (1 + v_expf(0 - x))`), then,
/// when `limit > 1e-6f`, `tmp = std::min(tmp, limit)` and `result =
/// std::max(-limit, std::min(limit, up))`; `result *= tmp`. `std::min(a, b)`
/// is `b < a ? b : a` and `std::max(a, b)` is `a < b ? b : a`.
fn ik_swiglu_clamp(gate: f32, up: f32, limit: f32) -> f32 {
    let mut tmp = gate / (1.0 + ik_expf(0.0 - gate));
    let mut result = up;
    if limit > 1e-6 {
        tmp = if limit < tmp { limit } else { tmp };
        result = if result < limit { result } else { limit };
        result = if -limit < result { result } else { -limit };
    }
    result * tmp
}

/// `swiglu_clamp` against the transcription, bit for bit, on inputs past the
/// limit on every side (silu above it, up above `limit` and below `-limit`,
/// the edges themselves, both zeros, NaN), in a vector whose length leaves an
/// eight-lane tail; every value alone must equal its value in the block. A
/// limit at or below 1e-6 (or NaN) clamps nothing: bit for bit `swiglu`.
#[test]
fn swiglu_clamp_matches_ik_rule() {
    let gates = [
        -200.0f32,
        -20.0,
        -10.5,
        -1.2785,
        -0.0,
        0.0,
        0.5,
        2.4,
        6.99,
        7.001,
        9.99,
        10.0,
        10.0005,
        10.001,
        10.5,
        12.0,
        30.0,
        88.0,
        100.0,
        200.0,
        f32::NAN,
    ];
    let ups = [
        -1000.0f32,
        -10.001,
        -10.0,
        -9.99,
        -7.5,
        -0.0,
        0.0,
        3.0,
        7.5,
        9.99,
        10.0,
        10.001,
        50.0,
        f32::NAN,
    ];
    let (gate, up): (Vec<f32>, Vec<f32>) = gates
        .iter()
        .flat_map(|&g| ups.iter().map(move |&u| (g, u)))
        .unzip();
    let n = gate.len();
    assert_ne!(n % 8, 0, "the set must leave a tail");
    let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    for limit in [10.0f32, 7.0, 2.5e-6] {
        let mut out = vec![0.0f32; n];
        qdot::swiglu_clamp(&gate, &up, limit, &mut out);
        let (mut silu_hi, mut up_hi, mut up_lo) = (0, 0, 0);
        for i in 0..n {
            let want = ik_swiglu_clamp(gate[i], up[i], limit);
            assert!(
                same(out[i], want),
                "limit {limit}: gate {} up {}: got {} ({:#x}), ik's rule {want} ({:#x})",
                gate[i],
                up[i],
                out[i],
                out[i].to_bits(),
                want.to_bits()
            );
            let mut one = [0.0f32];
            qdot::swiglu_clamp(&gate[i..i + 1], &up[i..i + 1], limit, &mut one);
            assert!(
                same(one[0], out[i]),
                "limit {limit}: i={i} moves with its position"
            );
            silu_hi += usize::from(gate[i] / (1.0 + ik_expf(0.0 - gate[i])) > limit);
            up_hi += usize::from(up[i] > limit);
            up_lo += usize::from(up[i] < -limit);
        }
        assert!(
            silu_hi > 0 && up_hi > 0 && up_lo > 0,
            "limit {limit}: the set must cross every side (silu > L {silu_hi}, u > L {up_hi}, u < -L {up_lo})"
        );
        eprintln!(
            "swiglu_clamp limit {limit}: {n} values bit-identical to ik's rule (silu > L {silu_hi}, u > L {up_hi}, u < -L {up_lo})"
        );
    }
    for limit in [1e-6f32, 0.0, -3.0, f32::NAN] {
        let (mut got, mut plain) = (vec![0.0f32; n], vec![0.0f32; n]);
        qdot::swiglu_clamp(&gate, &up, limit, &mut got);
        qdot::swiglu(&gate, &up, &mut plain);
        for i in 0..n {
            assert!(
                same(got[i], plain[i]),
                "limit {limit}: gate {} up {}: a limit <= 1e-6 must clamp nothing",
                gate[i],
                up[i]
            );
        }
    }
}

// ------------------------------------------------- SwiGLU-OAI (MiniMax-M3)
// The reference below is llama.cpp's `ggml_compute_forward_swiglu_oai_f32`
// (`ggml/src/ggml-cpu/ops.cpp`) written out again in scalar f32 with ik's
// exponential in place of libm's `expf`, not a call into qdot: a `+ 1` or an
// `alpha` deleted from `swiglu_oai` cannot be deleted from this too.

const OAI_ALPHA: f32 = 1.702;

/// `x = std::min(g, limit)`, `y = std::clamp(u, -limit, limit)`, `out_glu = x
/// / (1 + expf(alpha * (-x)))`, `out_glu * (y + 1)`. `std::min(a, b)` is `b <
/// a ? b : a`; `std::clamp(v, lo, hi)` is `v < lo ? lo : hi < v ? hi : v`.
fn ref_swiglu_oai(g: f32, u: f32, alpha: f32, limit: f32) -> f32 {
    let x = if limit < g { limit } else { g };
    let y = if u < -limit {
        -limit
    } else if limit < u {
        limit
    } else {
        u
    };
    let out_glu = x / (1.0 + ik_expf(alpha * -x));
    out_glu * (y + 1.0)
}

/// `swiglu_oai` against the transcription, bit for bit, on inputs past the
/// limit on every side (the gate above it, the up above and below its
/// bounds, the edges themselves, both zeros, a gate whose exponential
/// overflows), in a vector whose length leaves an eight-lane tail; every
/// value alone must equal its value in the block. And against the formula in
/// f64 — `x / (1 + e^z) · (y + 1)` at the f32 argument `z = alpha · (-x)` the
/// reference's `expf` takes — within the band the rounding sources give:
/// the exponential's 2 ulp (`ik_expf` sweeps to 1.84), the sum `1 + e`, the
/// quotient and the product half an ulp each, 3.5 · 2^-23 of the result,
/// beside the smallest normal for the values that underflow and the cliff where
/// the exponential overflows (`88.73 / alpha · (limit + 1) / f32::MAX`: the exact
/// value at the first `x` whose `e^z` is past `f32::MAX`, which both the vector
/// exponential and libm's flush to 0).
#[test]
fn swiglu_oai_matches_the_reference_rule() {
    // -52.1286 is the first gate whose exponential, `alpha · 52.1286` = 88.7232, is past f32::MAX.
    let gates = [
        -200.0f32, -60.0, -52.1286, -20.0, -10.5, -1.2785, -0.0, 0.0, 0.5, 2.4, 6.99, 7.0, 7.001,
        9.99, 30.0,
    ];
    let ups = [
        -1000.0f32, -10.001, -7.001, -7.0, -6.99, -1.0, -0.0, 0.0, 3.0, 6.99, 7.0, 7.001, 50.0,
    ];
    let (gate, up): (Vec<f32>, Vec<f32>) = gates
        .iter()
        .flat_map(|&g| ups.iter().map(move |&u| (g, u)))
        .unzip();
    let n = gate.len();
    assert_ne!(n % 8, 0, "the set must leave a tail");
    for limit in [7.0f32, 2.5, 0.0] {
        let mut out = vec![0.0f32; n];
        assert!(qdot::swiglu_oai(&gate, &up, OAI_ALPHA, limit, &mut out));
        let (mut gate_hi, mut up_hi, mut up_lo, mut worst) = (0, 0, 0, 0.0f64);
        for i in 0..n {
            let want = ref_swiglu_oai(gate[i], up[i], OAI_ALPHA, limit);
            assert_eq!(
                out[i].to_bits(),
                want.to_bits(),
                "limit {limit}: gate {} up {}: got {} ({:#x}), the reference rule {want} ({:#x})",
                gate[i],
                up[i],
                out[i],
                out[i].to_bits(),
                want.to_bits()
            );
            assert_eq!(
                qdot::swiglu_oai_1(gate[i], up[i], OAI_ALPHA, limit).to_bits(),
                want.to_bits()
            );
            let mut one = [0.0f32];
            assert!(qdot::swiglu_oai(
                &gate[i..=i],
                &up[i..=i],
                OAI_ALPHA,
                limit,
                &mut one
            ));
            assert_eq!(
                one[0].to_bits(),
                out[i].to_bits(),
                "limit {limit}: i={i} moves with its position"
            );
            gate_hi += usize::from(gate[i] > limit);
            up_hi += usize::from(up[i] > limit);
            up_lo += usize::from(up[i] < -limit);
            // The f64 twin of the formula at the f32 argument.
            let x = f64::from(gate[i].min(limit));
            let z = f64::from(OAI_ALPHA * -gate[i].min(limit));
            let y = f64::from(up[i]).clamp(-f64::from(limit), f64::from(limit));
            let exact = x / (1.0 + z.exp()) * (y + 1.0);
            let flush =
                88.73 / f64::from(OAI_ALPHA) * (f64::from(limit) + 1.0) / f64::from(f32::MAX);
            let band = 3.5 * 2f64.powi(-23) * exact.abs() + flush + f64::from(f32::MIN_POSITIVE);
            let d = (f64::from(out[i]) - exact).abs();
            worst = worst.max(d / band);
            assert!(
                d <= band,
                "limit {limit}: gate {} up {}: {} vs exact {exact}",
                gate[i],
                up[i],
                out[i]
            );
        }
        assert!(
            (limit == 0.0 || gate_hi > 0) && up_hi > 0 && up_lo > 0 && worst > 0.0,
            "limit {limit}: the set must cross every side (g > L {gate_hi}, u > L {up_hi}, u < -L {up_lo})"
        );
        eprintln!(
            "swiglu_oai limit {limit}: {n} values bit-identical to the reference rule, \
             max |d|/band {worst:.3} against the f64 formula"
        );
    }
}

/// The OAI combine reports a NaN or an infinity in `gate` or `up` at every
/// position (each lane of both vectors and the padded tail), which the clamps
/// would write as a finite value, and reports finite inputs as finite.
#[test]
fn swiglu_oai_flags_a_non_finite_input_at_every_position() {
    let n = 19;
    let mut out = vec![0.0f32; n];
    let base = vec![f32::MAX; n];
    assert!(qdot::swiglu_oai(&base, &base, OAI_ALPHA, 7.0, &mut out));
    for i in 0..n {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut v = base.clone();
            v[i] = bad;
            assert!(
                !qdot::swiglu_oai(&v, &base, OAI_ALPHA, 7.0, &mut out),
                "gate[{i}] = {bad}"
            );
            assert!(
                !qdot::swiglu_oai(&base, &v, OAI_ALPHA, 7.0, &mut out),
                "up[{i}] = {bad}"
            );
        }
    }
}

/// A limit the clamp of `up` cannot take (negative, NaN, infinite) is refused
/// by name, never run.
#[test]
fn swiglu_oai_refuses_a_limit_without_bounds() {
    for limit in [-1.0f32, f32::NAN, f32::INFINITY] {
        let r = std::panic::catch_unwind(|| {
            let mut out = [0.0f32];
            qdot::swiglu_oai(&[1.0], &[1.0], OAI_ALPHA, limit, &mut out)
        });
        assert!(r.is_err(), "limit {limit} was run");
    }
}

// --------------------------------------------------------- sum_sq_f64
// The laned f64 sum against the left-to-right one: equal to f64 rounding, and
// equal exactly once narrowed to f32 — the only form the norm consumes.
#[test]
fn sum_sq_f64_narrows_to_the_sequential_sum() {
    for n in [8usize, 512, 2048, 2048 + 5] {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 2001) as f32 - 1000.0) * 0.0137)
            .collect();
        let seq: f64 = x.iter().fold(0.0f64, |a, &v| a + (v * v) as f64);
        let got = qdot::sum_sq_f64(&x);
        assert!(
            ((got - seq) / seq).abs() < 1e-14,
            "n={n} got {got} seq {seq}"
        );
        assert_eq!(
            ((got / n as f64) as f32).to_bits(),
            ((seq / n as f64) as f32).to_bits()
        );
    }
}

// ------------------------------------------------------------ dot_f32
// The reference order by hand: eight lane sums (first block a multiply, the
// rest fused multiply-adds), then (l0+l4 + l2+l6) + (l1+l5 + l3+l7).
#[test]
fn dot_f32_is_the_reference_lane_order() {
    let k = 2048;
    let w: Vec<f32> = (0..k)
        .map(|i| ((i * 37 % 2001) as f32 - 1000.0) * 0.0013)
        .collect();
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 91 % 1777) as f32 - 888.0) * 0.0071)
        .collect();
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let got = qdot::dot_f32(&bytes, &x).expect("k is a multiple of 8");
    let mut lane = [0.0f32; 8];
    for (l, a) in lane.iter_mut().enumerate() {
        *a = x[l] * w[l];
    }
    for i in 1..k / 8 {
        for (l, a) in lane.iter_mut().enumerate() {
            *a = x[i * 8 + l].mul_add(w[i * 8 + l], *a);
        }
    }
    let s: [f32; 4] = std::array::from_fn(|l| lane[l] + lane[l + 4]);
    let want = (s[0] + s[2]) + (s[1] + s[3]);
    assert_eq!(got.to_bits(), want.to_bits());
    // A width the lanes cannot take is the caller's, not a truncated sum.
    assert!(qdot::dot_f32(&bytes[..4 * 12], &x[..12]).is_none());
}

// ------------------------------------------- quantize_col AVX2 encoders
// The activation encoders behind `quantize_col` run AVX2 when the CPU has it;
// the scalar loops stay as the fallback and as this section's oracle. Every
// test compares the dispatched `quantize_col` against `quantize_col_scalar`
// on the bytes, and pins hand-computed bytes wherever the value is derivable
// without borrowing either implementation.
//
// Contract <-> assertion table (q8_K = block_q8_K, q8_2 = block_q8_2_x4/tail):
//
// | contract (scalar semantics)                                    | assertion |
// |----------------------------------------------------------------|-----------|
// | q8_K: amax = max abs; max = FIRST value reaching it (strict >) | first-max ties: +/-m planted both orders, every one of the 8 lane positions, first and last 8-lane group, plus a third same-sign copy; d bytes checked against a hand first-scan |
// | q8_K: iscale = -127/max, plain f32 ops, no FMA; d = 1/iscale   | tie columns at max = +/-127 make iscale = -/+1.0 exact; d bytes and all 256 codes checked |
// | q8_K: code = min(127, nearest_int(iscale*v)) as i8             | products exactly n+0.5, even and odd n, both signs, both iscale signs; expected codes from the test-side round-ties-even oracle |
// | q8_K: amax == 0 -> the whole 296-byte block is zero            | all +0.0, all -0.0, zero blocks mixed with live ones (both block slots) |
// | q8_K layout: d@0..4, 0@4..8, codes@8..264, 16 i16 sums@264..296| every case compares the full byte range |
// | q8_2: d = bf16(amax/127), id = 1/d if d > 0 else 0             | amax ~1e-40 (bf16 underflows to 0 -> codes 0); amax ~1e-38 (smallest bf16 subnormal, 1/d overflows to inf -> codes 0); bf16 tie-round blocks assert d bits |
// | q8_2: code = nearest_int(v*id) as i8, NO clamp                 | tie columns with d = 1.0 and 2.0 exact (id = 1.0/0.5, products exact); expected codes from the same round-ties-even oracle |
// | q8_2 layout: 4 x [bf16 d @2ir, i16 isum @8+2ir, codes @16+32ir] in a zero-filled 144 B group; 36 B tail = [d, isum, codes] | shapes with 0 and 1-3 tail blocks; zero group/tail bytes checked directly |
// | byte identity on finite inputs                                | >= 200 pseudo-random columns per type per shape, magnitudes 1e-6..1e4, forced +/-max ties and zeros folded in |
// | out of scope: NaN inputs (scalar Rust max and vmaxps disagree on NaN ordering); the encoders' domain is finite f32 | none (domain statement) |

/// Deterministic LCG (Knuth MMIX) — the sweep must not grow a `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32) / 2147483648.0 - 1.0
    }
}

/// Round to nearest, ties to even — `nearest_int`'s semantics on the
/// encoders' |y| <= 127 domain, as an oracle independent of both encoders.
fn round_ties_even(v: f32) -> i32 {
    let r = v.round(); // ties away from zero; fix the half cases to even
    if (v - r).abs() == 0.5 && r % 2.0 != 0.0 {
        (r - v.signum()) as i32
    } else {
        r as i32
    }
}

/// The gates below compare `quantize_col`'s AVX2 path with the scalar mirror,
/// so on x86_64 they need AVX2. Off x86_64 `supports` holds through the
/// mirror, which is `quantize_col` there, and they hold it to their hand
/// oracles.
fn require_quantizer_avx2() {
    assert!(
        supports(GgmlType::Q3_K),
        "this gate compares quantize_col's AVX2 path against the scalar mirror; this CPU has no AVX2"
    );
}

/// Byte identity of the dispatched encoder against the scalar mirror, first
/// differing byte named in the panic.
fn assert_col_bit_identical(w: GgmlType, x: &[f32]) {
    let mut a = vec![0u8; col_bytes(w, x.len())];
    let mut b = vec![0u8; a.len()];
    quantize_col(w, x, &mut a);
    quantize_col_scalar(w, x, &mut b);
    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            p,
            q,
            "{w:?} k = {}: byte {i}: avx2 {p:02x} != scalar {q:02x}",
            x.len()
        );
    }
}

/// Sweep: 200 pseudo-random columns per type per shape over magnitudes
/// 1e-6..1e4, with forced +/-max ties (both signs at max magnitude — the
/// first-max pick decides) and zeros (both signs) folded in.
#[cfg(target_arch = "x86_64")]
#[test]
fn quantize_col_random_bit_identical() {
    require_quantizer_avx2();
    // Q4_K/Q6_K k is whole 256-value super-blocks (k_granularity); the x4
    // groups still tile the column 128 values at a time. Q5_0/Q5_1 take any
    // 32-multiple, including 1-3 tail blocks past the groups.
    let shapes: &[(GgmlType, &[usize])] = &[
        (GgmlType::Q3_K, &[256, 512, 2048]),
        (GgmlType::Q4_K, &[256, 512, 2048]),
        (GgmlType::Q6_K, &[256, 512, 2048]),
        (GgmlType::Q5_0, &[128, 160, 192, 352]),
        (GgmlType::Q5_1, &[128, 192, 10944]),
        (GgmlType::Q8_0, &[128, 224, 2144]),
        (GgmlType::IQ4_NL, &[128, 640, 672]),
        (GgmlType::IQ4_XS, &[256, 512]),
        (GgmlType::Q2_0, &[64, 256, 2048]),
    ];
    const COLS: usize = 200;
    const MAGS: [f32; 11] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 1e2, 1e3, 1e4];
    let mut rng = Lcg(0x5EED_0AAC_5EED);
    for &(w, ks) in shapes {
        for &k in ks {
            for c in 0..COLS {
                let mag = MAGS[c % MAGS.len()];
                let x: Vec<f32> = (0..k)
                    .map(|i| {
                        if i % 193 == 7 {
                            -0.0
                        } else if i % 97 == 3 {
                            0.0
                        } else if i % 89 == 11 {
                            mag
                        } else if i % 89 == 45 {
                            -mag
                        } else {
                            rng.unit() * mag
                        }
                    })
                    .collect();
                assert_col_bit_identical(w, &x);
            }
        }
    }
}

/// All-zero columns (both zero signs) and zero blocks beside live ones: the
/// q8_K zero block is 296 zero bytes; the q8_2 zero group is 144 and the
/// zero tail 36.
#[test]
fn quantize_col_zero_blocks_bit_identical() {
    require_quantizer_avx2();
    for fill in [0.0f32, -0.0] {
        assert_col_bit_identical(GgmlType::Q3_K, &vec![fill; 512]);
        assert_col_bit_identical(GgmlType::Q5_0, &vec![fill; 128]);
    }
    // q8_K: an all-zero middle block between live ones.
    let mut mixed = vec![0.25f32; 768];
    for v in mixed[256..512].iter_mut() {
        *v = -0.0;
    }
    assert_col_bit_identical(GgmlType::Q3_K, &mixed);
    let mut out = vec![0u8; col_bytes(GgmlType::Q3_K, 768)];
    quantize_col(GgmlType::Q3_K, &mixed, &mut out);
    assert!(
        out[296..592].iter().all(|&b| b == 0),
        "the all-zero middle q8_K block must be 296 zero bytes"
    );
    // q8_2: a zero group before live tails, and a zero tail after a live group.
    let mut x = vec![0.5f32; 192];
    for v in x[..128].iter_mut() {
        *v = 0.0;
    }
    assert_col_bit_identical(GgmlType::Q5_0, &x);
    let mut out = vec![0u8; col_bytes(GgmlType::Q5_0, 192)];
    quantize_col(GgmlType::Q5_0, &x, &mut out);
    assert!(
        out[..144].iter().all(|&b| b == 0),
        "the all-zero group must be 144 zero bytes"
    );
    let mut x = vec![0.5f32; 160];
    for v in x[128..].iter_mut() {
        *v = -0.0;
    }
    assert_col_bit_identical(GgmlType::Q5_1, &x);
    let mut out = vec![0u8; col_bytes(GgmlType::Q5_1, 160)];
    quantize_col(GgmlType::Q5_1, &x, &mut out);
    assert!(
        out[144..].iter().all(|&b| b == 0),
        "the all-zero tail must be 36 zero bytes"
    );
}

/// q8_K first-max ties: the signed `max` is the value of the FIRST element
/// reaching amax. A wrong pick flips iscale's sign with it, and with it every
/// code in the block — planted ties make that loud. Both super-blocks of a
/// k = 512 column carry the pattern so both block slots are covered.
#[test]
fn quantize_col_q8k_first_max_ties() {
    require_quantizer_avx2();
    let k = 512usize;
    let mut rng = Lcg(0xAB57_F1F1_0001);
    for lane in 0..8usize {
        for g8 in [0usize, 31] {
            for plus_first in [true, false] {
                let mut x: Vec<f32> = (0..k).map(|_| rng.unit() * 0.4).collect();
                for blk in [0usize, 256] {
                    let p = blk + 8 * g8 + lane;
                    let q = blk + 8 * (31 - g8) + (7 - lane);
                    let mid = blk + 128 + lane;
                    let (first, last) = if plus_first {
                        (3.0f32, -3.0)
                    } else {
                        (-3.0, 3.0)
                    };
                    x[p] = first;
                    x[q] = last;
                    x[mid] = first;
                }
                // Hand oracle: the scalar's sequential strict-> scan.
                let mut amax = 0.0f32;
                let mut max = 0.0f32;
                for &v in &x {
                    let ax = v.abs();
                    if ax > amax {
                        amax = ax;
                        max = v;
                    }
                }
                let expected_d = 1.0f32 / (-127.0f32 / max);
                let mut out = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
                quantize_col(GgmlType::Q3_K, &x, &mut out);
                assert_eq!(
                    out[0..4],
                    expected_d.to_le_bytes(),
                    "lane {lane} group {g8} plus_first {plus_first}: d must come from the FIRST max (picked {max})"
                );
                assert_col_bit_identical(GgmlType::Q3_K, &x);
            }
        }
    }
}

/// Rounding ties: columns built so iscale (resp. id) is exactly -/+1.0 or
/// 0.5 and the products land exactly on n+0.5 — even and odd n, both signs.
#[test]
fn quantize_col_rounding_ties() {
    require_quantizer_avx2();
    let ties: [f32; 19] = [
        0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 0.25, 7.5, -7.5, 63.5, -63.5, 120.5, -120.5,
        0.0, -0.0, 127.0, -127.0,
    ];

    // q8_K: first max +/-127 -> iscale = -/+1.0 exact -> y = -+x exactly.
    for first in [127.0f32, -127.0] {
        let mut x: Vec<f32> = Vec::with_capacity(256);
        x.push(first);
        for i in 1..256 {
            x.push(ties[i % ties.len()]);
        }
        let iscale = -127.0f32 / first;
        let mut out = vec![0u8; col_bytes(GgmlType::Q3_K, 256)];
        quantize_col(GgmlType::Q3_K, &x, &mut out);
        for (j, &v) in x.iter().enumerate() {
            let e = round_ties_even(v * iscale).min(127) as i8;
            assert_eq!(out[8 + j], e as u8, "first max {first}: code {j} (v = {v})");
        }
        assert_col_bit_identical(GgmlType::Q3_K, &x);
    }

    // q8_2 over four blocks (one 144 B group): d = 1.0 (id = 1.0), d = 2.0
    // (id = 0.5, values doubled so products are the same exact ties), and two
    // bf16-conversion ties — 1+2^-8 rounds down to d = 1.0 (even kept lsb),
    // 1+3·2^-8 rounds up to d = 1+2^-6 (odd kept lsb).
    let mut x: Vec<f32> = Vec::with_capacity(128);
    // 127*(1+2^-8) and 127*(1+3*2^-8): exact f32, both on bf16 tie boundaries.
    let amaxes = [
        127.0f32,
        254.0,
        127.0 + 127.0 / 256.0,
        127.0 + 3.0 * 127.0 / 256.0,
    ];
    for (b, &amax) in amaxes.iter().enumerate() {
        let scale = if b == 1 { 2.0 } else { 1.0 };
        x.push(amax);
        for i in 1..32 {
            x.push(ties[i % ties.len()] * scale);
        }
    }
    let mut out = vec![0u8; col_bytes(GgmlType::Q5_0, 128)];
    quantize_col(GgmlType::Q5_0, &x, &mut out);
    for ir in 0..4 {
        let d = u16::from_le_bytes([out[2 * ir], out[2 * ir + 1]]);
        let expect_d = match ir {
            0 | 2 => 0x3F80, // bf16 1.0
            1 => 0x4000,     // bf16 2.0
            _ => 0x3F82,     // bf16 1+2^-8 (tie rounded up)
        };
        assert_eq!(d, expect_d, "block {ir}: bf16 scale bits");
        if ir < 3 {
            // id = 1.0 for blocks 0/2, 0.5 for block 1: products are the ties.
            let id = if ir == 1 { 0.5f32 } else { 1.0 };
            let mut isum = 0i32;
            for j in 0..32 {
                let v = x[32 * ir + j];
                let e = round_ties_even(v * id) as i8;
                assert_eq!(
                    out[16 + 32 * ir + j],
                    e as u8,
                    "block {ir} code {j} (v = {v})"
                );
                isum += e as i32;
            }
            assert_eq!(
                i16::from_le_bytes([out[8 + 2 * ir], out[9 + 2 * ir]]),
                isum as i16,
                "block {ir}: isum"
            );
        }
    }
    assert_col_bit_identical(GgmlType::Q5_0, &x);
}

/// The saturating end of both encoders. For q8_K, |fl(iscale)*v| <=
/// 127*(1+2^-24)^2 < 127.5, so `nearest_int` never reaches 128 and the
/// one-sided clamp stays dormant in-domain — the columns below pin the
/// reachable extremes: codes at exactly -127/+127. For q8_2 the same extreme
/// through the bf16 scale, including the tie that rounds amax/127 UP to d=1.
#[test]
fn quantize_col_saturation_ends() {
    require_quantizer_avx2();
    for (m, first_sign) in [
        (0.00390625f32, 1.0f32),
        (0.00390625, -1.0),
        (127.0, 1.0),
        (127.0, -1.0),
    ] {
        let x: Vec<f32> = (0..256)
            .map(|i| {
                if i == 0 {
                    m * first_sign
                } else if i % 2 == 0 {
                    m
                } else {
                    -m
                }
            })
            .collect();
        let iscale = -127.0f32 / (m * first_sign);
        let mut out = vec![0u8; col_bytes(GgmlType::Q3_K, 256)];
        quantize_col(GgmlType::Q3_K, &x, &mut out);
        let mut seen = (false, false);
        for (j, &v) in x.iter().enumerate() {
            let e = qdot::nearest_int(iscale * v).min(127) as i8;
            assert_eq!(out[8 + j], e as u8, "m = {m} first {first_sign}: code {j}");
            seen.0 |= e == -127;
            seen.1 |= e == 127;
        }
        assert!(
            seen.0 && seen.1,
            "m = {m}: both code extremes must be reached"
        );
        assert_col_bit_identical(GgmlType::Q3_K, &x);
    }

    // q8_2: alternating +-amax blocks hit codes -/+127 (d = 2.0 gives
    // id = 0.5 exactly on values 2*127); 127*(1-2^-9) rounds amax/127 UP to
    // d = 1, id = 1.0, and the codes still reach the extreme.
    let mut x: Vec<f32> = Vec::with_capacity(128);
    let amp = 254.0f32;
    let v126_75 = 127.0f32 - 127.0 / 512.0; // 127*(1-2^-9), exact in f32
    for a in [amp, amp, amp, v126_75] {
        for j in 0..32 {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            x.push(a * sign);
        }
    }
    let mut out = vec![0u8; col_bytes(GgmlType::Q5_0, 128)];
    quantize_col(GgmlType::Q5_0, &x, &mut out);
    for ir in 0..4 {
        let d = u16::from_le_bytes([out[2 * ir], out[2 * ir + 1]]);
        let expect_d = if ir < 3 { 0x4000 } else { 0x3F80 };
        assert_eq!(d, expect_d, "block {ir}: bf16 scale bits");
        let mut seen = (false, false);
        for j in 0..32 {
            let code = out[16 + 32 * ir + j] as i8;
            seen.0 |= code == -127;
            seen.1 |= code == 127;
        }
        assert!(
            seen.0 && seen.1,
            "block {ir}: both code extremes must be reached"
        );
    }
    assert_col_bit_identical(GgmlType::Q5_0, &x);
}

/// Subnormal and tiny amax: the q8_2 scale's bf16 round-trip, and the iscale
/// overflow of a subnormal-magnitude q8_K block. In the id = inf band every
/// product is ±inf or NaN and the codes must come out 0 exactly like the
/// scalar's mantissa-form nearest_int — a saturating convert would emit -1.
#[test]
fn quantize_col_subnormal_tiny_scales() {
    require_quantizer_avx2();
    // amax ~1e-40: amax/127 rounds to bf16 0 -> d == 0 -> id == 0 -> zero codes.
    let mut x = vec![1e-40f32; 128];
    x[0] = -1e-40;
    assert_col_bit_identical(GgmlType::Q5_0, &x);
    let mut out = vec![0u8; col_bytes(GgmlType::Q5_0, 128)];
    quantize_col(GgmlType::Q5_0, &x, &mut out);
    assert!(
        out.iter().all(|&b| b == 0),
        "bf16 scale underflowed to 0: the group must be all zero"
    );

    // amax ~1e-38: amax/127 rounds to the smallest bf16 subnormal (bits
    // 0x0001); 1/d overflows to inf, so every product is ±inf or NaN and the
    // wrapped mantissa-form round must give code 0 and isum 0, like scalar.
    let x2 = vec![1e-38f32; 128];
    assert_col_bit_identical(GgmlType::Q5_0, &x2);
    let mut out2 = vec![0u8; col_bytes(GgmlType::Q5_0, 128)];
    quantize_col(GgmlType::Q5_0, &x2, &mut out2);
    for ir in 0..4 {
        assert_eq!(
            &out2[2 * ir..2 * ir + 2],
            &[0x01, 0x00][..],
            "block {ir}: d is the smallest bf16 subnormal"
        );
    }
    assert!(
        out2[8..].iter().all(|&b| b == 0),
        "inf/NaN products must quantize to 0 codes and 0 isums"
    );

    // All three bands in one column: blocks must not bleed state.
    let mut x3 = vec![0.0f32; 128];
    x3[..32].fill(1e-40);
    x3[32..64].fill(1e-38);
    x3[64..].fill(0.5);
    assert_col_bit_identical(GgmlType::Q5_0, &x3);

    // q8_K subnormal amax: iscale = -127/1e-38 overflows to -inf, so d = -0.0
    // and every product is ±inf or NaN -> all codes 0, like scalar.
    let x4 = vec![1e-38f32; 256];
    assert_col_bit_identical(GgmlType::Q3_K, &x4);
    let mut out4 = vec![0u8; col_bytes(GgmlType::Q3_K, 256)];
    quantize_col(GgmlType::Q3_K, &x4, &mut out4);
    assert_eq!(
        &out4[0..4],
        &(-0.0f32).to_le_bytes()[..],
        "d = 1/-inf = -0.0"
    );
    assert!(
        out4[4..].iter().all(|&b| b == 0),
        "inf/NaN products must quantize to 0 codes and 0 sums"
    );

    // q8_0 (Q2_0's column): at amax 1e-38, below 127 / f32::MAX, the multiplier 127 / amax
    // overflows, so d = 0 and the codes flush to 0; at 1e-36 it is finite, d still rounds to
    // an f16 zero and every code is 127.
    let mut x5 = vec![1e-38f32; 128];
    x5[64..].fill(1e-36);
    assert_col_bit_identical(GgmlType::Q2_0, &x5);
    let mut out5 = vec![0u8; col_bytes(GgmlType::Q2_0, 128)];
    quantize_col(GgmlType::Q2_0, &x5, &mut out5);
    assert!(
        out5[..68].iter().all(|&b| b == 0),
        "an overflowing multiplier must flush the blocks to zero"
    );
    for blk in out5[68..].as_chunks::<34>().0 {
        assert!(
            blk[..2] == [0, 0] && blk[2..].iter().all(|&c| c == 127),
            "a finite multiplier codes amax as 127 under an f16-zero d"
        );
    }
}

/// A NaN or an infinity in an activation block is undefined input, and both
/// encoder paths refuse it with the same named panic instead of quantizing the
/// block to a defined value (the AVX2 lane-wise max would otherwise skip a NaN
/// or drop a lane's maximum behind one, and an infinity would give a zero
/// scale). The cases are the ones a lane-wise max gets wrong: an all-NaN block,
/// a NaN that is the last value of its lane, a NaN between a lane's block
/// maximum and a later, smaller value of the same lane; and one +inf block.
#[test]
fn quantize_col_non_finite_panics_on_both_paths() {
    require_quantizer_avx2();
    let base = |k: usize| -> Vec<f32> { (0..k).map(|i| ((i % 29) as f32 - 14.0) / 16.0).collect() };
    // (type, k, block width the encoder scans for its max)
    for (w, k, blk) in [
        (GgmlType::Q3_K, 512, 256),
        (GgmlType::Q4_K, 256, 32),
        (GgmlType::Q5_0, 160, 32),
        (GgmlType::Q2_0, 128, 32),
    ] {
        let mut all_nan = base(k);
        all_nan[..blk].fill(f32::NAN);
        let mut last_in_lane = base(k);
        last_in_lane[blk - 3] = f32::NAN;
        // Lane 5 of the block: the maximum at group 0, NaN at group 1, a
        // smaller finite value at every later group.
        let mut dropped_max = base(k);
        dropped_max[5] = 40.0;
        dropped_max[8 + 5] = f32::NAN;
        let mut inf = base(k);
        inf[blk / 2] = f32::INFINITY;
        let mut tail_nan = base(k);
        tail_nan[k - blk..].fill(f32::NAN);
        for (name, x) in [
            ("all_nan", &all_nan),
            ("last_in_lane", &last_in_lane),
            ("dropped_max", &dropped_max),
            ("inf", &inf),
            ("tail_nan", &tail_nan),
        ] {
            for (path, f) in [
                ("avx2", quantize_col as fn(GgmlType, &[f32], &mut [u8])),
                (
                    "scalar",
                    quantize_col_scalar as fn(GgmlType, &[f32], &mut [u8]),
                ),
            ] {
                let mut out = vec![0u8; col_bytes(w, k)];
                let r =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(w, x, &mut out)));
                let msg = match r {
                    Ok(()) => panic!(
                        "{w:?} {name} {path}: quantized a non-finite block instead of panicking"
                    ),
                    Err(e) => e
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_default(),
                };
                assert!(
                    msg.contains("non-finite activation"),
                    "{w:?} {name} {path}: panicked, but not with the named message: {msg:?}"
                );
            }
        }
    }
}

/// Every f16 block scale through the kernels and their mirrors: the kernels decode
/// scales with F16C, the mirrors with `half_to_f32`, and the two converters must agree
/// on every f16 but a NaN (whose payload F16C may quiet) — subnormal and extreme scales
/// included, which real rows rarely carry. One block per type, the scale fields swept
/// over all 65,536 bit patterns, every second field with its sign flipped so a kernel
/// that reads one field in another's place (Q4_K's `dmin` for `d`) differs, the rest
/// fixed pseudo-random bytes. `dot_row` must
/// give the AVX2 kernel's bits too: it runs Q4_K's AVX-512 VNNI body where the CPU
/// has one ([`qdot::q4k_avx512`]).
#[cfg(target_arch = "x86_64")]
#[test]
fn f16_scales_bit_identical_kernel_vs_mirror() {
    let mut rng = Lcg(0x00F1_6C5C_A1E5);
    // (type, block bytes, offsets of its f16 scale fields)
    for (w, block, fields) in [
        (GgmlType::Q2_K, 84usize, &[80usize, 82][..]),
        (GgmlType::Q3_K, 110, &[108][..]),
        (GgmlType::Q4_K, 144, &[0, 2][..]),
        (GgmlType::Q5_K, 176, &[0, 2][..]),
        (GgmlType::Q6_K, 210, &[208][..]),
        (GgmlType::IQ2_XS, 74, &[0][..]),
        (GgmlType::IQ3_XXS, 98, &[0][..]),
        (GgmlType::IQ3_S, 110, &[0][..]),
        // IQ4_NL at k = 256 is eight 18-byte blocks, one f16 scale each.
        (
            GgmlType::IQ4_NL,
            144,
            &[0, 18, 36, 54, 72, 90, 108, 126][..],
        ),
        // IQ4_XS at k = 256 is one 136-byte block.
        (GgmlType::IQ4_XS, 136, &[0][..]),
        // Q8_0 at k = 256 is eight 34-byte blocks, one f16 scale each.
        (
            GgmlType::Q8_0,
            272,
            &[0, 34, 68, 102, 136, 170, 204, 238][..],
        ),
        // Q2_0 at k = 256 is four 18-byte blocks.
        (GgmlType::Q2_0, 72, &[0, 18, 36, 54][..]),
    ] {
        assert!(supports(w), "{w:?}: this CPU lacks the kernel's ISA");
        let mut row: Vec<u8> = (0..block).map(|_| rng.next_u32() as u8).collect();
        let x: Vec<f32> = (0..256).map(|_| rng.unit()).collect();
        let mut acol = vec![0u8; col_bytes(w, 256)];
        quantize_col(w, &x, &mut acol);
        for h in 0..=u16::MAX {
            if (h >> 10) & 0x1f == 0x1f && h & 0x3ff != 0 {
                continue; // NaN: only the payload may differ
            }
            for (j, &f) in fields.iter().enumerate() {
                let hj = h ^ ((j as u16 & 1) << 15);
                row[f..f + 2].copy_from_slice(&hj.to_le_bytes());
            }
            let a = dot_row_avx2(w, &row, &acol, 256).unwrap();
            let b = dot_row_scalar(w, &row, &acol, 256).unwrap();
            let c = dot_row(w, &row, &acol, 256).unwrap();
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{w:?} scale {h:#06x}: kernel {a:e} (bits {:#x}) vs mirror {b:e} (bits {:#x})",
                a.to_bits(),
                b.to_bits()
            );
            assert!(
                c.to_bits() == a.to_bits() || (c.is_nan() && a.is_nan()),
                "{w:?} scale {h:#06x}: dot_row {c:e} (bits {:#x}) vs kernel {a:e} (bits {:#x})",
                c.to_bits(),
                a.to_bits()
            );
        }
    }
}

// ------------------------------------------------ q8_2 code range (bf16 scale)

/// ik's x86 q8_2_x4 rule for one 32-value block (`quantize_row_q8_1_x4_T` with
/// `block_q8_2`): `d = bf16(amax / 127)` rounded to nearest even, `id = 1/d`
/// (0 when `d` is 0), each code `round_half_even(v * id)` converted to i32 and
/// packed to i8 with signed saturation (`_mm256_packs_epi32/16`), the sum the
/// unsaturated i32 codes' taken to i16. `None` where `1/d` overflows, which ik
/// turns into integer-indefinite codes and this crate's encoders flush to zero.
fn ik_q82_block(x: &[f32]) -> Option<([i8; 32], u16, i16)> {
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let u = (amax / 127.0).to_bits();
    let t = (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16;
    let d = f32::from_bits(u32::from(t) << 16);
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    if !id.is_finite() {
        return None;
    }
    let mut codes = [0i8; 32];
    let mut isum = 0i32;
    for (c, &v) in codes.iter_mut().zip(x) {
        let q = (v * id).round_ties_even() as i32;
        *c = q.clamp(-128, 127) as i8;
        isum = isum.wrapping_add(q);
    }
    Some((codes, t, isum as i16))
}

/// The qdot gates' one q8_2_x4 reader (`common::restore_q82x4`) against the rule's round trip, on every shape of block count:
/// groups only, tails only, both. Its tail arm is read by no oracle case (their `k` is a whole number of groups), so this
/// is the clause that pins it.
#[test]
fn restore_q82x4_reads_groups_and_tails() {
    for nb in [1usize, 3, 4, 5, 7, 9] {
        let k = 32 * nb;
        // Blocks of different magnitude, so a block read with another's scale is a wrong value.
        let x: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.37).sin() * (1.0 + (i / 32) as f32))
            .collect();
        let mut col = vec![0u8; col_bytes(GgmlType::IQ4_NL, k)];
        quantize_col(GgmlType::IQ4_NL, &x, &mut col);
        let mut rt = vec![0.0f32; k];
        quantize_row_q8_2_x4_roundtrip(&x, &mut rt);
        let restored = restore_q82x4(&col);
        assert_eq!(restored.len(), k, "{nb} blocks");
        for (i, (r, w)) in restored.iter().zip(&rt).enumerate() {
            assert_eq!(*r, f64::from(*w), "{nb} blocks, value {i}");
        }
    }
}

/// The q8_2 encoders (scalar and AVX2) against ik's rule where the bf16 scale
/// rounds down hardest. For a normal scale the round-down is at most 2^-8
/// relative — at a binade's bottom with a tie, `amax = 127·(1 + 2^-8)·2^e`,
/// `d = 2^e` and `amax·id = 127.49609375` — so every code stays in ±127: the
/// wrap the scalar's `as i8` would do is unreachable there, swept over every
/// exponent a normal scale takes. Below that, where `amax / 127` is a
/// subnormal (`amax < 127·2^-126`), bf16 keeps fewer bits: at `2^-127` the
/// round-down reaches 2^-7, `amax·id` reaches 127.99 and rounds to 128, which
/// ik saturates to 127 — the encoders must too, not wrap it to -128. Where
/// `1/d` overflows (`amax/127` at or below about `2^-128`) the encoders flush
/// the block to zero codes and a zero sum (ik's integer-indefinite codes carry
/// no value either); that regime is asserted as the flush.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_q82_codes_at_bf16_round_down() {
    assert!(
        supports(GgmlType::Q5_0),
        "the gate compares the AVX2 encoder with the scalar one; this CPU has no AVX2"
    );
    // Normal scales: the tie at every binade bottom, then a dense random
    // sweep of amax values.
    let mut normal: Vec<f32> = (-126..=120)
        .map(|e| 127.0 * (1.0 + 2f32.powi(-8)) * 2f32.powi(e))
        .collect();
    let mut s = 0x1234_5678u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    for _ in 0..4000 {
        // Random exponent in the normal-scale range, random mantissa.
        let e = (next() % 246) as i32 - 126;
        let m = 1.0 + (next() >> 8) as f32 / (1u32 << 24) as f32;
        normal.push(127.0 * m * 2f32.powi(e));
    }
    // Subnormal scales with a finite inverse: amax / 127 in [2^-127, 2^-126).
    let lo = 2f32.powi(-127);
    let mut subnormal = vec![127.0 * lo * (1.0 + 2f32.powi(-7))];
    for _ in 0..2000 {
        let f = (next() >> 8) as f32 / (1u32 << 24) as f32;
        subnormal.push(127.0 * lo * (1.0 + f));
    }
    // Overflowing inverses.
    let flush: Vec<f32> = [2f32.powi(-128), 2f32.powi(-130), 2f32.powi(-133)]
        .iter()
        .map(|&a| 127.0 * a)
        .collect();

    // Each amax becomes one block: ±amax at the ends, fractions between.
    let block = |amax: f32| -> [f32; 32] {
        std::array::from_fn(|i| match i {
            0 => amax,
            1 => -amax,
            2 => 0.0,
            _ => amax * ((i as f32 - 16.0) / 16.5),
        })
    };
    let mut checked = [0usize; 3];
    let mut max_code = 0i32;
    for (regime, amaxes) in [&normal, &subnormal, &flush].into_iter().enumerate() {
        for chunk in amaxes.chunks(5) {
            // Five blocks: one x4 group and one tail block (Q5_0's columns
            // are whole 32-value blocks; its activation is the same q8_2_x4
            // encoder as Q4_K's).
            let x: Vec<f32> = chunk.iter().flat_map(|&a| block(a)).collect();
            let k = x.len();
            let mut avx = vec![0u8; col_bytes(GgmlType::Q5_0, k)];
            let mut sca = vec![0u8; col_bytes(GgmlType::Q5_0, k)];
            quantize_col(GgmlType::Q5_0, &x, &mut avx);
            quantize_col_scalar(GgmlType::Q5_0, &x, &mut sca);
            assert_eq!(
                avx, sca,
                "regime {regime}: AVX2 and scalar bytes differ at {chunk:?}"
            );
            for (b, got) in q82_blocks(&avx, chunk.len()).into_iter().enumerate() {
                let xb = &x[32 * b..32 * b + 32];
                match (regime, ik_q82_block(xb)) {
                    (2, None) => {
                        assert!(
                            got.0.iter().all(|&c| c == 0) && got.2 == 0,
                            "amax {:e}: an overflowing inverse must flush the block, got {got:?}",
                            chunk[b]
                        );
                    }
                    (_, Some(want)) => {
                        assert_eq!(got, want, "regime {regime}: amax {:e}", chunk[b]);
                        if regime == 0 {
                            let m = got.0.iter().map(|&c| i32::from(c).abs()).max().unwrap_or(0);
                            assert!(m <= 127, "amax {:e}: code {m} past 127", chunk[b]);
                            max_code = max_code.max(m);
                        }
                    }
                    (r, None) => panic!("regime {r}: amax {:e} overflows its inverse", chunk[b]),
                }
                checked[regime] += 1;
            }
        }
    }
    println!(
        "q8_2 code range: {} normal-scale blocks (max |code| {max_code}), {} subnormal-scale blocks \
         saturated as ik's, {} overflowing-inverse blocks flushed; AVX2 bytes == scalar bytes",
        checked[0], checked[1], checked[2]
    );
}

/// The q8_K encoders (scalar and AVX2) at every scale an activation block can
/// have, down through the subnormals. Their scale is the f32 `iscale =
/// -127/max` itself, not a narrowed copy, so where `iscale` is finite every
/// product `iscale·v` with `|v| <= |max|` stays within `127·(1 + 2^-24)² <
/// 127.5` and rounds into ±127: the `.min(127)` clamp and the `as i8` wrap
/// (`v_wrap_i8` on the AVX2 side) are never reached. That is asserted on the
/// unclamped `nearest_int` of every value, over every exponent from the
/// largest normal down to the smallest subnormal, with random mantissas and
/// the values around `127 / f32::MAX`, where `iscale` starts to overflow.
/// Where it overflows (every subnormal `max` among them) the encoders flush
/// the block: codes 0, sums 0 — asserted as the flush. AVX2 bytes == scalar
/// bytes throughout.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_q8k_codes_at_subnormal_scales() {
    use qdot::nearest_int;
    assert!(
        supports(GgmlType::Q3_K),
        "the gate compares the AVX2 encoder with the scalar one; this CPU has no AVX2"
    );
    let mut s = 0x9abc_def1u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    // 2^e from its bits, exact for every f32 power of two, normal or
    // subnormal (`powi` may form 2^-e first, which overflows below 2^-127).
    let pow2 = |e: i32| -> f32 {
        if e >= -126 {
            f32::from_bits(((e + 127) as u32) << 23)
        } else {
            f32::from_bits(1u32 << (e + 149))
        }
    };
    // Every binade's bottom, then random mantissas at random exponents,
    // normal and subnormal (f32's powers of two run 2^-149 .. 2^127).
    let mut amaxes: Vec<f32> = (-149..=127).map(pow2).collect();
    for _ in 0..6000 {
        let e = (next() % 277) as i32 - 149;
        let m = 1.0 + (next() >> 8) as f32 / (1u32 << 24) as f32;
        let a = m * pow2(e);
        assert!(a.is_finite() && a > 0.0, "amax {m} * 2^{e} is {a}");
        amaxes.push(a);
    }
    // Around the edge where -127/max overflows.
    let edge = 127.0 / f32::MAX;
    let mut up = edge;
    let mut down = edge;
    for _ in 0..64 {
        amaxes.push(up);
        amaxes.push(down);
        up = f32::from_bits(up.to_bits() + 1);
        down = f32::from_bits(down.to_bits() - 1);
    }

    // One block per amax: max first (its sign alternates by block), its
    // negation, a zero, then fractions of it that land on every code.
    let block = |amax: f32, neg: bool| -> [f32; 256] {
        let max = if neg { -amax } else { amax };
        std::array::from_fn(|i| match i {
            0 => max,
            1 => -max,
            2 => 0.0,
            _ => max * ((i as f32 - 129.0) / 127.5),
        })
    };
    let (mut finite, mut flushed, mut max_code) = (0usize, 0usize, 0i32);
    for (c, chunk) in amaxes.chunks(4).enumerate() {
        let x: Vec<f32> = chunk
            .iter()
            .enumerate()
            .flat_map(|(b, &a)| block(a, (c + b) % 2 == 1))
            .collect();
        let k = x.len();
        let mut avx = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
        let mut sca = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
        quantize_col(GgmlType::Q3_K, &x, &mut avx);
        quantize_col_scalar(GgmlType::Q3_K, &x, &mut sca);
        assert_eq!(avx, sca, "AVX2 and scalar bytes differ at amax {chunk:?}");
        for (b, &amax) in chunk.iter().enumerate() {
            let xb = &x[256 * b..256 * (b + 1)];
            let ob = &avx[296 * b..296 * (b + 1)];
            let codes = &ob[8..264];
            let sums: Vec<i16> = ob[264..296]
                .chunks(2)
                .map(|p| i16::from_le_bytes([p[0], p[1]]))
                .collect();
            let iscale = -127.0f32 / xb[0];
            if !iscale.is_finite() {
                assert!(
                    codes.iter().all(|&q| q == 0) && sums.iter().all(|&v| v == 0),
                    "amax {amax:e}: an overflowing iscale must flush the block"
                );
                flushed += 1;
                continue;
            }
            for (i, &v) in xb.iter().enumerate() {
                let q = nearest_int(iscale * v);
                assert!(
                    (-127..=127).contains(&q),
                    "amax {amax:e}: value {i} ({v:e}) rounds to {q}, past ±127 before the clamp"
                );
                assert_eq!(codes[i] as i8, q as i8, "amax {amax:e}: value {i} code");
                max_code = max_code.max(q.abs());
            }
            for (j, &got) in sums.iter().enumerate() {
                let want: i32 = codes[16 * j..16 * j + 16]
                    .iter()
                    .map(|&q| i32::from(q as i8))
                    .sum();
                assert_eq!(i32::from(got), want, "amax {amax:e}: sum {j}");
            }
            finite += 1;
        }
    }
    let pass = finite > 0 && flushed > 0 && max_code == 127;
    println!(
        "q8_K code range: {finite} finite-iscale blocks (max |code| {max_code}, no value past ±127 \
         before the clamp), {flushed} overflowing-iscale blocks flushed; AVX2 bytes == scalar bytes"
    );
    assert!(
        pass,
        "the sweep must reach both regimes and code 127 (finite {finite}, flushed {flushed}, \
         max |code| {max_code})"
    );
}

// ------------------------------------------------- multi-column tiles
// `dot_row_cols` against `dot_row` per column, bit for bit, for every column
// count 1..=TILE_COLS and every tile slot a column can sit in, on real rows
// of the V4.1 host-expert types and on columns that reach the quantizer's
// ends. The bit contract holds by construction — the same integer products
// per block, the same float order per column — so any differing value is a
// bug, not rounding.

/// Rows of the first `ty` tensor of the V4.1 split set whose name contains
/// `prefer` (the host expert stack of that type), from the first shard on
/// that carries one — the first shard holds only block 0, whose routed down
/// is Q5_K — read at `data_base + offset` from the header-only inventory
/// (the strict `Gguf::open` refuses the first shard's bf16 `token_embd`):
/// (name, k, row bytes, the first `rows` rows).
fn v41_rows(ty: GgmlType, prefer: &str, rows: usize) -> (String, usize, usize, Vec<u8>) {
    let block = match ty {
        GgmlType::Q3_K => 110,
        GgmlType::Q4_K => 144,
        GgmlType::Q5_K => 176,
        GgmlType::Q6_K => 210,
        _ => panic!("no tile row source for {ty:?}"),
    };
    split_rows(&gguf::v41::model(), ty, block, 256, prefer, rows)
}

/// [`v41_rows`] for any split set named by one of its shards (`first`, or the
/// only file): the type's `block` bytes per `gran` values.
fn split_rows(
    first: &str,
    ty: GgmlType,
    block: usize,
    gran: usize,
    prefer: &str,
    rows: usize,
) -> (String, usize, usize, Vec<u8>) {
    use std::os::unix::fs::FileExt;
    let shards: Vec<String> = match first.find("-00001-of-") {
        Some(at) => (1..=99)
            .map(|i| format!("{}-{i:05}-of-{}", &first[..at], &first[at + 10..]))
            .take_while(|p| std::path::Path::new(p).exists())
            .collect(),
        None => vec![first.to_string()],
    };
    for path in &shards {
        let inv = gguf::inventory_of(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let Some(t) = inv
            .tensors
            .iter()
            .find(|t| t.type_id == ty.as_u32() && t.name.contains(prefer))
        else {
            continue;
        };
        let k = t.dims[0] as usize;
        assert!(k.is_multiple_of(gran), "{}: k = {k}", t.name);
        let n = t.dims[1..].iter().product::<u64>() as usize;
        let row_bytes = block * k / gran;
        assert_eq!(t.nbytes, Some((n * row_bytes) as u64), "{}: bytes", t.name);
        assert!(
            n >= rows,
            "{}: only {n} rows, the gate needs {rows}",
            t.name
        );
        let mut bytes = vec![0u8; rows * row_bytes];
        std::fs::File::open(path)
            .and_then(|f| f.read_exact_at(&mut bytes, inv.data_base + t.offset))
            .unwrap_or_else(|e| panic!("{path}: read {}: {e}", t.name));
        return (t.name.clone(), k, row_bytes, bytes);
    }
    panic!(
        "no shard of {first} ({} found) carries a {ty:?} tensor named *{prefer}*",
        shards.len()
    );
}

/// Rows each tile clause dots.
const TILE_ROWS: usize = 256;

/// The seeded activation columns of a tile clause, `k` values each: seven
/// random columns over magnitudes 1e-3..1e2 with outliers every 61st value,
/// a column of zeros, and a column at the quantizer's ends (±3.0
/// alternating: codes ±127 in both q8_K and q8_2_x4).
fn tile_columns(k: usize) -> Vec<Vec<f32>> {
    let mut rng = Lcg(0x7117_c01d);
    let mut cols: Vec<Vec<f32>> = [1e-3f32, 0.02, 0.3, 1.0, 4.0, 7.5, 1e2]
        .iter()
        .map(|&mag| {
            (0..k)
                .map(|i| {
                    let u = rng.unit() * mag;
                    if i % 61 == 0 { u * 8.0 } else { u }
                })
                .collect()
        })
        .collect();
    cols.push(vec![0.0; k]);
    cols.push(
        (0..k)
            .map(|i| if i % 2 == 0 { -3.0 } else { 3.0 })
            .collect(),
    );
    cols
}

/// Clause of one row set: for c = 1..=TILE_COLS and every starting column
/// of the cyclic list `acols`, every row's `dot_row_cols` over the c columns
/// from there equals `dot_row` per column, bit for bit. The outputs start as
/// NaN, so an unwritten one fails too. Returns the calls made.
fn assert_tile_matches(
    ty: GgmlType,
    label: &str,
    k: usize,
    row_bytes: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
) -> usize {
    assert!(
        qdot::has_tile(ty),
        "{label}: {ty:?} has no tile kernel on this CPU — the clause would compare dot_row with itself"
    );
    let rows = bytes.len() / row_bytes;
    let n = acols.len();
    assert!(
        n >= qdot::TILE_COLS,
        "{label}: {n} columns cannot fill a tile"
    );
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for (j, a) in acols.iter().enumerate() {
            want[r * n + j] = dot_row(ty, src, a, k).unwrap();
        }
    }
    let mut calls = 0;
    for c in 1..=qdot::TILE_COLS {
        for s in 0..n {
            let idx: Vec<usize> = (0..c).map(|i| (s + i) % n).collect();
            let cols: Vec<&[u8]> = idx.iter().map(|&j| acols[j].as_slice()).collect();
            for r in 0..rows {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                let mut out = vec![f32::NAN; c];
                qdot::dot_row_cols(ty, src, &cols, k, &mut out).unwrap();
                for (slot, (&j, &got)) in idx.iter().zip(&out).enumerate() {
                    let w = want[r * n + j];
                    assert_eq!(
                        got.to_bits(),
                        w.to_bits(),
                        "{label}: row {r}, c = {c}, slot {slot} (column {j}): tile {got:e} \
                         (bits {:#x}) vs dot_row {w:e} (bits {:#x})",
                        got.to_bits(),
                        w.to_bits()
                    );
                }
                calls += 1;
            }
        }
    }
    calls
}

/// The seeded columns of [`tile_columns`] coded for `ty`, after `extra`
/// already-coded ones.
fn coded_columns(ty: GgmlType, k: usize, extra: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut cols = extra;
    for x in tile_columns(k) {
        let mut a = vec![0u8; col_bytes(ty, k)];
        quantize_col(ty, &x, &mut a);
        cols.push(a);
    }
    cols
}

/// A synthetic Q3_K super-block: code `i` stores `u(i)` in 0..7 (its value
/// plus 4 — bit 2 is the high bit), 16-code block `j` the scale `s(j)` in
/// −32..31, and the f16 bits `d`; packed as the scalar mirror reads a block.
fn q3k_block(u: impl Fn(usize) -> u8, s: impl Fn(usize) -> i8, d: u16) -> [u8; 110] {
    let mut b = [0u8; 110];
    for i in 0..256 {
        let v = u(i);
        assert!(v < 8, "code {i}: u = {v}");
        let (half, field, cell) = (i / 128, (i / 32) % 4, i % 32);
        if v & 4 != 0 {
            b[cell] |= 1 << (4 * half + field);
        }
        b[32 + 32 * half + cell] |= (v & 3) << (2 * field);
    }
    for j in 0..16 {
        let sv = u8::try_from(i32::from(s(j)) + 32)
            .ok()
            .filter(|&v| v < 64)
            .unwrap_or_else(|| panic!("block {j}: scale {} outside -32..31", s(j)));
        if j < 8 {
            b[96 + j] |= sv & 0xF;
        } else {
            b[96 + j - 8] |= (sv & 0xF) << 4;
        }
        b[104 + j % 4] |= (sv >> 4) << (2 * (j / 4));
    }
    b[108..110].copy_from_slice(&d.to_le_bytes());
    b
}

/// A synthetic q8_K column of `k` values in `quantize_col`'s block layout:
/// code `i` is `q(i)`, every block's scale `d`, and each block's 16 sums of
/// 16 codes at bytes 264..296.
fn q8k_col(k: usize, q: impl Fn(usize) -> i8, d: f32) -> Vec<u8> {
    let mut a = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
    for (sb, blk) in a.as_chunks_mut::<296>().0.iter_mut().enumerate() {
        blk[0..4].copy_from_slice(&d.to_le_bytes());
        for j in 0..16 {
            let mut sum = 0i32;
            for l in 0..16 {
                let v = q(256 * sb + 16 * j + l);
                blk[8 + 16 * j + l] = v as u8;
                sum += i32::from(v);
            }
            let sum = i16::try_from(sum).expect("16 i8 codes sum within i16");
            blk[264 + 2 * j..266 + 2 * j].copy_from_slice(&sum.to_le_bytes());
        }
    }
    a
}

/// A synthetic Q3_K super-block's codes `u(i)` and scales `s(j)`, as
/// [`q3k_block`] takes them.
type Q3kKind = (fn(usize) -> u8, fn(usize) -> i8);

/// The synthetic ends both Q3_K tile clauses start from, at the ends of the
/// tiles' folded terms: seven block kinds — codes u = 0 and u = 7 (values −4
/// and 3), scales −32 and 31 (uniform, alternating by 16-code block, and a
/// ramp) — and four q8_K columns of `k` values: every code −127 (block sums
/// −2032), +127, −128 (outside the encoder's range, still exact) and ±127
/// alternating by block.
fn q3k_ends(k: usize) -> ([Q3kKind; 7], Vec<Vec<u8>>) {
    let kinds: [Q3kKind; 7] = [
        (|_| 0, |_| -32),
        (|_| 0, |_| 31),
        (|_| 7, |_| -32),
        (|_| 7, |_| 31),
        (|_| 0, |j| if j % 2 == 0 { -32 } else { 31 }),
        (|_| 7, |j| if j % 2 == 0 { 31 } else { -32 }),
        (|i| (i % 8) as u8, |j| 4 * j as i8 - 32),
    ];
    let cols = vec![
        q8k_col(k, |_| -127, 0.01),
        q8k_col(k, |_| 127, 0.01),
        q8k_col(k, |_| -128, 0.01),
        q8k_col(k, |i| if (i / 16) % 2 == 0 { -127 } else { 127 }, 0.02),
    ];
    (kinds, cols)
}

/// Q3_K tile clause: rows and columns at the ends of the tile's folded
/// terms first ([`q3k_ends`]), then the V4.1 first shard's routed gate stack
/// (k = the model width), then V2-Lite's `ffn_gate`-1 rows on the oracle's
/// six `ffn_norm-1` tokens (the rows and columns gate 1's harness reads) —
/// tile = `dot_row` per column, bit for bit.
#[test]
#[ignore = "hw: needs the box, the V4.1 shard, the model file and $BLOOMERY_DATA/ref"]
fn hw_q3k_tile_matches_dot_row() {
    let ty = GgmlType::Q3_K;
    // Rows of two super-blocks, d = 1.0 then 0.5.
    let k = 512;
    let (kinds, ends) = q3k_ends(k);
    let mut rows = Vec::new();
    for (u, s) in kinds {
        for d in [0x3C00u16, 0x3800] {
            rows.extend_from_slice(&q3k_block(u, s, d));
        }
    }
    let calls = assert_tile_matches(
        ty,
        "synthetic ends",
        k,
        k / 256 * 110,
        &rows,
        &coded_columns(ty, k, ends),
    );
    eprintln!("q3k tile: synthetic ends (k = {k}): {calls} row calls, c = 1..=8, bit-identical");

    let (name, k, row_bytes, bytes) = v41_rows(ty, "ffn_gate_exps", TILE_ROWS);
    let calls = assert_tile_matches(
        ty,
        &name,
        k,
        row_bytes,
        &bytes,
        &coded_columns(ty, k, vec![]),
    );
    eprintln!("q3k tile: {name} (k = {k}): {calls} row calls, c = 1..=8, bit-identical");

    let g = gguf::Gguf::open(model_path()).unwrap();
    let t = g
        .find("blk.1.ffn_gate_exps.weight")
        .or_else(|| g.find("blk.0.ffn_gate.weight"));
    let t = t.expect("V2-Lite carries a Q3_K ffn_gate");
    assert_eq!(t.ty, ty, "{}: type", t.name);
    let k2 = t.dims[0] as usize;
    let rb2 = k2 / 256 * 110;
    let rows2 = &g.data(t).unwrap()[..TILE_ROWS * rb2];
    let toks = oracle_f32("ffn_norm-1", k2 * 6);
    let real: Vec<Vec<u8>> = toks
        .chunks_exact(k2)
        .map(|x| {
            let mut a = vec![0u8; col_bytes(ty, k2)];
            quantize_col(ty, x, &mut a);
            a
        })
        .collect();
    let calls = assert_tile_matches(ty, &t.name, k2, rb2, rows2, &coded_columns(ty, k2, real));
    eprintln!(
        "q3k tile: {} (k = {k2}) on six oracle tokens + seeded: {calls} calls, bit-identical",
        t.name
    );
}

/// Q4_K tile clause: the V2-Lite Q4_K tensor the gate-triple harness reads,
/// on ik's dumped column and the seeded ones, then the V4.1 set's first Q4_K
/// routed down stack — tile = `dot_row` per column, bit for bit.
#[test]
#[ignore = "hw: needs the box, the model file, the V4.1 shards and $BLOOMERY_DATA/ref"]
fn hw_q4k_tile_matches_dot_row() {
    let ty = GgmlType::Q4_K;
    // PIN(2026-10-14): the tile clause on q4k_synthetic_rows, which
    // hw_q4k_avx512_matches_avx2_synthetic also reads.
    let (block, _) = lane_block(ty);
    for (label, k, bytes) in q4k_synthetic_rows() {
        assert_tile_matches(ty, &label, k, k / 256 * block, &bytes, &lane_columns(ty, k));
    }
    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| w.ty == ty && (w.dims[0] as usize).is_multiple_of(256))
        .expect("the model must carry at least one aligned Q4_K tensor");
    let k = w.dims[0] as usize;
    let row_bytes = 144 * k / 256;
    let bytes = &g.data(w).unwrap()[..TILE_ROWS * row_bytes];
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q4k-x4-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, _) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    let calls = assert_tile_matches(
        ty,
        &w.name,
        k,
        row_bytes,
        bytes,
        &coded_columns(ty, k, vec![ik_acol]),
    );
    eprintln!(
        "q4k tile: {} (k = {k}) on ik's column + seeded: {calls} calls, bit-identical",
        w.name
    );

    let (name, k, row_bytes, bytes) = v41_rows(ty, "ffn_down_exps", TILE_ROWS);
    let calls = assert_tile_matches(
        ty,
        &name,
        k,
        row_bytes,
        &bytes,
        &coded_columns(ty, k, vec![]),
    );
    eprintln!("q4k tile: {name} (k = {k}): {calls} calls, bit-identical");
}

/// Q5_K tile clause: the V4.1 first shard's routed down stack on ik's dumped
/// column (the Q5_K harness's) and the seeded ones — tile = `dot_row` per
/// column, bit for bit.
#[test]
#[ignore = "hw: needs the box, the V4.1 shard and $BLOOMERY_DATA/ref"]
fn hw_q5k_tile_matches_dot_row() {
    let ty = GgmlType::Q5_K;
    let (name, k, row_bytes, bytes) = v41_rows(ty, "ffn_down_exps", TILE_ROWS);
    let c = load_q5k(1);
    assert_eq!(
        c.k, k,
        "the Q5_K harness's tensor and the down stack share k"
    );
    let (ik_acol, _) = q5k_ik_dump(&c);
    let calls = assert_tile_matches(
        ty,
        &name,
        k,
        row_bytes,
        &bytes,
        &coded_columns(ty, k, vec![ik_acol]),
    );
    eprintln!("q5k tile: {name} (k = {k}) on ik's column + seeded: {calls} calls, bit-identical");
}

/// Row lengths of the Q5_1 and Q8_0 tile clauses: Qwen3.8's routed down
/// (k = 640, five x4 groups), its gate and up (2560), 736 — five groups and
/// three q8_2 tail blocks — and 96, three tail blocks and no group.
const LEGACY_TILE_KS: [usize; 4] = [640, 2560, 736, 96];

/// Random rows per length in a Q5_1 or Q8_0 tile clause.
const LEGACY_RANDOM_ROWS: usize = 64;

/// f16 bits of a finite value in ±[2^-10, 2^3), either sign: the scale draw
/// of the random legacy rows.
fn rand_f16(rng: &mut Lcg) -> u16 {
    let r = rng.next_u32();
    let sign = ((r >> 31) as u16) << 15;
    let exp = (5 + (r >> 10) % 13) as u16;
    sign | (exp << 10) | (r & 0x3FF) as u16
}

/// A synthetic Q5_1 block: code `i` stores `u(i)` in 0..31 (bit 4 in qh bit
/// `i`), the f16 bits `d` and `m`; packed as the scalar mirror reads a block.
fn q5f1_block(u: impl Fn(usize) -> u8, d: u16, m: u16) -> [u8; 24] {
    let mut b = [0u8; 24];
    b[0..2].copy_from_slice(&d.to_le_bytes());
    b[2..4].copy_from_slice(&m.to_le_bytes());
    let mut qh = 0u32;
    for i in 0..32 {
        let v = u(i);
        assert!(v < 32, "code {i}: u = {v}");
        qh |= u32::from(v >> 4) << i;
        if i < 16 {
            b[8 + i] |= v & 0xF;
        } else {
            b[8 + i - 16] |= (v & 0xF) << 4;
        }
    }
    b[4..8].copy_from_slice(&qh.to_le_bytes());
    b
}

/// A synthetic Q8_0 block: code `i` is `q(i)`, the f16 bits `d`.
fn q8f0_block(q: impl Fn(usize) -> i8, d: u16) -> [u8; 34] {
    qdot::Q8Block {
        d,
        q: std::array::from_fn(q),
    }
    .to_bytes()
}

/// The ends a Q5_1 tile clause's adversarial rows are built from, as (codes,
/// d, m) block kinds: the max code 31 under a positive and a negative min,
/// code 0 under both signs, codes alternating 0 and 31 with m = 0, a ramp
/// at d = 0 (the min term alone), and a block of d = m = 0.
type Q5f1Kind = (fn(usize) -> u8, u16, u16);
const Q5F1_ENDS: [Q5f1Kind; 7] = [
    (|_| 31, 0x3C00, 0x4000),
    (|_| 31, 0x3C00, 0xC000),
    (|_| 0, 0x3C00, 0xC000),
    (|_| 0, 0x3800, 0x4000),
    (|i| if i % 2 == 0 { 0 } else { 31 }, 0x3C00, 0x0000),
    (|i| (i % 32) as u8, 0x0000, 0xBC00),
    (|_| 31, 0x0000, 0x0000),
];

/// The ends a Q8_0 tile clause's adversarial rows are built from, as (codes,
/// d) block kinds: codes 127, −127 and −128 (whose `sign(w, w)` stays −128,
/// read unsigned as 128), ±127/−128 alternating under a negative scale, a
/// ramp over the whole code range, a block of d = 0, and a subnormal scale.
type Q8f0Kind = (fn(usize) -> i8, u16);
const Q8F0_ENDS: [Q8f0Kind; 7] = [
    (|_| 127, 0x3C00),
    (|_| -127, 0x3C00),
    (|_| -128, 0x3C00),
    (|i| if i % 2 == 0 { 127 } else { -128 }, 0xBC00),
    (|i| (8 * i) as u8 as i8, 0x3800),
    (|_| 127, 0x0000),
    (|i| i as i8 - 16, 0x0001),
];

/// Rows of `nb` blocks from `kinds` (`block(kind)` packs one): one row per
/// kind whose block `b` takes kind `(r + b) % n` — every kind in every x4
/// slot — then one row per kind of that kind alone.
fn legacy_end_rows<K: Copy>(kinds: &[K], nb: usize, block: impl Fn(K) -> Vec<u8>) -> Vec<u8> {
    let n = kinds.len();
    let mut rows = Vec::new();
    for r in 0..n {
        for b in 0..nb {
            rows.extend_from_slice(&block(kinds[(r + b) % n]));
        }
    }
    for &kind in kinds {
        for _ in 0..nb {
            rows.extend_from_slice(&block(kind));
        }
    }
    rows
}

/// The columns of a Q5_1 or Q8_0 tile clause: every value +3.0 (codes 127,
/// block sums +4064), every value −3.0 (sums −4064), then [`tile_columns`]'
/// nine, after `extra` already-coded ones.
fn legacy_tile_columns(ty: GgmlType, k: usize, extra: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut cols = extra;
    for v in [3.0f32, -3.0] {
        let mut a = vec![0u8; col_bytes(ty, k)];
        quantize_col(ty, &vec![v; k], &mut a);
        cols.push(a);
    }
    coded_columns(ty, k, cols)
}

/// Q5_1 tile clause: at each of [`LEGACY_TILE_KS`], random rows (codes,
/// d and m of either sign) and then the ends of [`Q5F1_ENDS`], against the
/// ±3.0 columns and the seeded ones; then V2-Lite's Q5_1 `ffn_down` (k =
/// 10944: two tail blocks) — tile = `dot_row` per column, bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2) and the model file"]
fn hw_q5f1_tile_matches_dot_row() {
    let ty = GgmlType::Q5_1;
    let mut rng = Lcg(0x51_7113);
    for k in LEGACY_TILE_KS {
        let nb = k / 32;
        let row_bytes = nb * 24;
        let cols = legacy_tile_columns(ty, k, vec![]);
        let mut rows = Vec::new();
        for _ in 0..LEGACY_RANDOM_ROWS * nb {
            let (d, m) = (rand_f16(&mut rng), rand_f16(&mut rng));
            let codes: [u8; 32] = std::array::from_fn(|_| (rng.next_u32() % 32) as u8);
            rows.extend_from_slice(&q5f1_block(|i| codes[i], d, m));
        }
        let calls = assert_tile_matches(ty, "random rows", k, row_bytes, &rows, &cols);
        eprintln!("q5_1 tile: random rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
        let rows = legacy_end_rows(&Q5F1_ENDS, nb, |(u, d, m)| q5f1_block(u, d, m).to_vec());
        let calls = assert_tile_matches(ty, "end rows", k, row_bytes, &rows, &cols);
        eprintln!("q5_1 tile: end rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
    }

    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| w.ty == ty && (w.dims[0] as usize).is_multiple_of(32))
        .expect("the model must carry the Q5_1 tensor (blk.0.ffn_down)");
    let k = w.dims[0] as usize;
    let row_bytes = k / 32 * 24;
    let bytes = &g.data(w).unwrap()[..TILE_ROWS * row_bytes];
    let calls = assert_tile_matches(
        ty,
        &w.name,
        k,
        row_bytes,
        bytes,
        &legacy_tile_columns(ty, k, vec![]),
    );
    eprintln!(
        "q5_1 tile: {} (k = {k}): {calls} calls, c = 1..=8, bit-identical",
        w.name
    );
}

/// Q8_0 tile clause: at each of [`LEGACY_TILE_KS`], random rows (every code,
/// d of either sign) and then the ends of [`Q8F0_ENDS`], against the ±3.0
/// columns and the seeded ones; then the ik harness's 64 rows (k = 2144:
/// three tail blocks) with ik's own column, whose block 9 carries a −128
/// code under a subnormal scale — the sign trick's wrap — tile = `dot_row`
/// per column, bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2) and $BLOOMERY_DATA/ref"]
fn hw_q8f0_tile_matches_dot_row() {
    let ty = GgmlType::Q8_0;
    let mut rng = Lcg(0x80_7113);
    for k in LEGACY_TILE_KS {
        let nb = k / 32;
        let row_bytes = nb * 34;
        let cols = legacy_tile_columns(ty, k, vec![]);
        let mut rows = Vec::new();
        for _ in 0..LEGACY_RANDOM_ROWS * nb {
            let d = rand_f16(&mut rng);
            let codes: [i8; 32] = std::array::from_fn(|_| rng.next_u32() as i8);
            rows.extend_from_slice(&q8f0_block(|i| codes[i], d));
        }
        let calls = assert_tile_matches(ty, "random rows", k, row_bytes, &rows, &cols);
        eprintln!("q8_0 tile: random rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
        let rows = legacy_end_rows(&Q8F0_ENDS, nb, |(q, d)| q8f0_block(q, d).to_vec());
        let calls = assert_tile_matches(ty, "end rows", k, row_bytes, &rows, &cols);
        eprintln!("q8_0 tile: end rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
    }

    let d = q8f0_dump();
    let rows = d.rows.concat();
    let calls = assert_tile_matches(
        ty,
        "ik's rows",
        d.k,
        34 * d.k / 32,
        &rows,
        &legacy_tile_columns(ty, d.k, vec![d.ik_acol.clone()]),
    );
    eprintln!(
        "q8_0 tile: ik's rows on ik's column + seeded (k = {}): {calls} calls, c = 1..=8, \
         bit-identical",
        d.k
    );
}

// ------------------------------------- IQ3_XXS, IQ4_XS and IQ4_NL tiles
// The Qwen3.8 UD-Q3_K_XL routed experts: IQ3_XXS and IQ4_XS gate/up
// (k = 2560, Q8_K columns), IQ4_NL down (k = 640, Q8_2_X4 columns). Each clause
// runs random rows, end rows and the file's real rows through
// `assert_tile_matches`.

/// A kind of end block: its bytes, whole.
type IqEnd = fn() -> Vec<u8>;

/// f16 bits of 1.0, −1.0 and the smallest subnormal.
const F16_ONE: u16 = 0x3C00;
const F16_NEG_ONE: u16 = 0xBC00;
const F16_TINY: u16 = 0x0001;

/// An IQ3_XXS block: f16 `d`, the 64 grid index bytes from `idx`, the eight
/// sign/scale words from `word`.
fn iq3xxs_block(d: u16, idx: impl Fn(usize) -> u8, word: impl Fn(usize) -> u32) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    b.extend((0..64).map(idx));
    for w in 0..8 {
        b.extend_from_slice(&word(w).to_le_bytes());
    }
    b
}

/// An IQ4_XS block: f16 `d`, `scales_h`, `scales_l` and the 128 code bytes
/// from `qs`.
fn iq4xs_block(d: u16, scales_h: u16, scales_l: [u8; 4], qs: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    b.extend_from_slice(&scales_h.to_le_bytes());
    b.extend_from_slice(&scales_l);
    b.extend((0..128).map(qs));
    b
}

/// An IQ4_NL block: f16 `d` and the 16 code bytes from `qs`.
fn iq4nl_block(d: u16, qs: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut b = d.to_le_bytes().to_vec();
    b.extend((0..16).map(qs));
    b
}

/// IQ3_XXS end blocks: the largest grid entries with every sign bit and the
/// top scale under either sign of `d`, the smallest scale with no signs, a
/// scrambled pattern, d = 0 and a subnormal d.
const IQ3XXS_ENDS: [IqEnd; 7] = [
    || iq3xxs_block(F16_ONE, |_| 0xFF, |_| 0xFFFF_FFFF),
    || iq3xxs_block(F16_NEG_ONE, |_| 0xFF, |_| 0xFFFF_FFFF),
    || iq3xxs_block(F16_ONE, |_| 0xFF, |_| 0),
    || iq3xxs_block(F16_ONE, |_| 0, |_| 0x0FFF_FFFF),
    || {
        iq3xxs_block(
            F16_ONE,
            |i| (i * 37 + 11) as u8,
            |w| 0xA5C3_1E87u32.rotate_left(5 * w as u32),
        )
    },
    || iq3xxs_block(0, |_| 0xFF, |_| 0xFFFF_FFFF),
    || iq3xxs_block(F16_TINY, |i| i as u8, |_| 0xF0F0_F0F0),
];

/// IQ4_XS end blocks: every code 15 (offset 241, whose pair with a ±127
/// column byte saturates `maddubs`) under the top and the bottom scales, every
/// code 0, half-and-half nibbles, a ramp, d = 0 and a subnormal d.
const IQ4XS_ENDS: [IqEnd; 8] = [
    || iq4xs_block(F16_ONE, 0xFFFF, [0xFF; 4], |_| 0xFF),
    || iq4xs_block(F16_NEG_ONE, 0xFFFF, [0xFF; 4], |_| 0xFF),
    || iq4xs_block(F16_ONE, 0, [0; 4], |_| 0xFF),
    || iq4xs_block(F16_ONE, 0xFFFF, [0xFF; 4], |_| 0),
    || iq4xs_block(F16_ONE, 0xA5A5, [0x5A, 0xA5, 0x3C, 0xC3], |_| 0x0F),
    || {
        iq4xs_block(F16_ONE, 0x1234, [0x12, 0x34, 0x56, 0x78], |i| {
            (i * 37) as u8
        })
    },
    || iq4xs_block(0, 0xFFFF, [0xFF; 4], |_| 0xFF),
    || iq4xs_block(F16_TINY, 0xFFFF, [0xFF; 4], |_| 0xF0),
];

/// IQ4_NL end blocks: every code 15 and every code 0 (|w| 113 and 127),
/// alternating nibbles, a ramp, a negative d, d = 0 and a subnormal d.
const IQ4NL_ENDS: [IqEnd; 7] = [
    || iq4nl_block(F16_ONE, |_| 0xFF),
    || iq4nl_block(F16_ONE, |_| 0),
    || iq4nl_block(F16_ONE, |_| 0x0F),
    || iq4nl_block(F16_ONE, |i| (i * 37) as u8),
    || iq4nl_block(F16_NEG_ONE, |_| 0xF0),
    || iq4nl_block(0, |_| 0xFF),
    || iq4nl_block(F16_TINY, |i| i as u8),
];

/// A random block of `len` bytes whose f16 scales (at the `fields` offsets) are
/// finite and of either sign.
fn iq_random_block(rng: &mut Lcg, len: usize, fields: &[usize]) -> Vec<u8> {
    let mut b: Vec<u8> = (0..len).map(|_| (rng.next_u32() >> 13) as u8).collect();
    for &f in fields {
        b[f..f + 2].copy_from_slice(&rand_f16(rng).to_le_bytes());
    }
    b
}

/// Random rows and end rows of `ty` at each of `ks`, against the ±3.0 and
/// seeded columns: tile = `dot_row` per column, bit for bit. `block` bytes
/// hold `gran` values.
fn iq_tile_synthetic(ty: GgmlType, block: usize, gran: usize, ks: &[usize], ends: &[IqEnd]) {
    let mut rng = Lcg(0x1A_7113 ^ block as u64);
    let fields = f16_scale_fields(ty);
    for &k in ks {
        let nb = k / gran;
        let row_bytes = nb * block;
        let cols = legacy_tile_columns(ty, k, vec![]);
        let mut rows = Vec::new();
        for _ in 0..LEGACY_RANDOM_ROWS * nb {
            rows.extend_from_slice(&iq_random_block(&mut rng, block, fields));
        }
        let calls = assert_tile_matches(ty, "random rows", k, row_bytes, &rows, &cols);
        eprintln!("{ty:?} tile: random rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
        let rows = legacy_end_rows(ends, nb, |end| end());
        let calls = assert_tile_matches(ty, "end rows", k, row_bytes, &rows, &cols);
        eprintln!("{ty:?} tile: end rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
    }
}

/// The first [`TILE_ROWS`] rows of the Qwen3.8 UD-Q3_K_XL routed-expert
/// tensor of `ty` against the seeded columns: tile = `dot_row` per column.
fn iq_tile_real(ty: GgmlType, block: usize, gran: usize) {
    // The first shard of the file `tools/ref/build-qdot-ref.sh` names for the dumps.
    let first =
        "/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf";
    let (name, k, row_bytes, bytes) = split_rows(first, ty, block, gran, "exps", TILE_ROWS);
    let calls = assert_tile_matches(
        ty,
        &name,
        k,
        row_bytes,
        &bytes,
        &legacy_tile_columns(ty, k, vec![]),
    );
    eprintln!("{ty:?} tile: {name} (k = {k}): {calls} calls, c = 1..=8, bit-identical");
}

/// IQ3_XXS tile clause: random rows and the ends of [`IQ3XXS_ENDS`] at k =
/// 2560 (gate/up), 768 and 256, then the file's first routed-expert rows —
/// tile = `dot_row` per column, bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2) and the Qwen3.8 UD-Q3_K_XL file"]
fn hw_iq3xxs_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ3_XXS, 98, 256, &[2560, 768, 256], &IQ3XXS_ENDS);
    iq_tile_real(GgmlType::IQ3_XXS, 98, 256);
}

/// IQ4_XS tile clause: random rows and the ends of [`IQ4XS_ENDS`] (the code-15
/// blocks reach the `maddubs` saturation against the ±3.0 columns) at k =
/// 2560, 768 and 256, then the file's first routed-expert rows — tile =
/// `dot_row` per column, bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2) and the Qwen3.8 UD-Q3_K_XL file"]
fn hw_iq4xs_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ4_XS, 136, 256, &[2560, 768, 256], &IQ4XS_ENDS);
    iq_tile_real(GgmlType::IQ4_XS, 136, 256);
}

/// IQ3_S tile clause: random rows and the ends of [`IQ3S_ENDS`] at k = 4096 (gate/up), 768
/// and 256 — tile = `dot_row` per column, bit for bit. No IQ3_S file is on the box, so there
/// are no real rows.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq3s_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ3_S, 110, 256, &[4096, 768, 256], &IQ3S_ENDS);
}

/// Q2_K tile clause: random rows (every scale and min nibble, code and `d`/`dmin` of either
/// sign) and the ends of [`Q2K_ENDS`] at k = 4096, 768 and 256 — tile = `dot_row` per column,
/// bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_q2k_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::Q2_K, 84, 256, &[4096, 768, 256], &Q2K_ENDS);
}

/// IQ2_XS tile clause: random rows (every grid index, sign index and scale nibble, `d` of
/// either sign) and the ends of [`IQ2XS_ENDS`] at k = 4096, 768 and 256 — tile = `dot_row`
/// per column, bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq2xs_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ2_XS, 74, 256, &[4096, 768, 256], &IQ2XS_ENDS);
}

/// IQ2_XXS tile clause: random rows and the ends of [`IQ2XXS_ENDS`] at k = 4096, 768 and
/// 256 — tile = `dot_row` per column, bit for bit. No IQ2_XXS file is on the box, so
/// there are no real rows.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq2xxs_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ2_XXS, 66, 256, &[4096, 768, 256], &IQ2XXS_ENDS);
}

/// Clause of one row set: for c = 1..=WIDE_COLS and every starting column of
/// the cyclic list `acols`, every row's `dot_row_cols_wide` over the c columns
/// from there equals `dot_row` per column, bit for bit — c past the list's
/// length repeats its columns, so the eight-column groups after the first
/// read the same columns at other slots. The outputs start as NaN, so an
/// unwritten one fails too. Returns the calls made.
fn assert_wide_matches(
    ty: GgmlType,
    label: &str,
    k: usize,
    row_bytes: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
) -> usize {
    assert!(
        qdot::has_wide(ty),
        "{label}: {ty:?} has no wide kernel on this CPU — the clause would compare dot_row_cols \
         with dot_row"
    );
    let rows = bytes.len() / row_bytes;
    let n = acols.len();
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for (j, a) in acols.iter().enumerate() {
            want[r * n + j] = dot_row(ty, src, a, k).unwrap();
        }
    }
    let mut calls = 0;
    for c in 1..=qdot::WIDE_COLS {
        for s in 0..n {
            let idx: Vec<usize> = (0..c).map(|i| (s + i) % n).collect();
            let cols: Vec<&[u8]> = idx.iter().map(|&j| acols[j].as_slice()).collect();
            for r in 0..rows {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                let mut out = vec![f32::NAN; c];
                qdot::dot_row_cols_wide(ty, src, &cols, k, &mut out).unwrap();
                for (slot, (&j, &got)) in idx.iter().zip(&out).enumerate() {
                    let w = want[r * n + j];
                    assert_eq!(
                        got.to_bits(),
                        w.to_bits(),
                        "{label}: row {r}, c = {c}, slot {slot} (column {j}): wide {got:e} \
                         (bits {:#x}) vs dot_row {w:e} (bits {:#x})",
                        got.to_bits(),
                        w.to_bits()
                    );
                }
                calls += 1;
            }
        }
    }
    calls
}

/// IQ2_XXS wide clause: random rows (every grid index, sign index and scale nibble, `d` of
/// either sign) and the ends of [`IQ2XXS_ENDS`] at k = 4096 (Muse Spark's experts), 768 and
/// 256, against the ±3.0 and seeded columns — `dot_row_cols_wide` = `dot_row` per column, bit
/// for bit, at every column count 1..=WIDE_COLS.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_iq2xxs_wide_matches_dot_row() {
    let ty = GgmlType::IQ2_XXS;
    let (block, gran) = (66, 256);
    let mut rng = Lcg(0x31DE_1A2E);
    let fields = f16_scale_fields(ty);
    for k in [4096, 768, 256] {
        let nb = k / gran;
        let row_bytes = nb * block;
        let cols = legacy_tile_columns(ty, k, vec![]);
        let mut rows = Vec::new();
        for _ in 0..LEGACY_RANDOM_ROWS * nb {
            rows.extend_from_slice(&iq_random_block(&mut rng, block, fields));
        }
        let calls = assert_wide_matches(ty, "random rows", k, row_bytes, &rows, &cols);
        eprintln!(
            "{ty:?} wide: random rows (k = {k}): {calls} calls, c = 1..={}, bit-identical",
            qdot::WIDE_COLS
        );
        let rows = legacy_end_rows(&IQ2XXS_ENDS, nb, |end| end());
        let calls = assert_wide_matches(ty, "end rows", k, row_bytes, &rows, &cols);
        eprintln!(
            "{ty:?} wide: end rows (k = {k}): {calls} calls, c = 1..={}, bit-identical",
            qdot::WIDE_COLS
        );
    }
}

/// A wide call's column count outside `1..=WIDE_COLS`, an `out` of another length and a short
/// column are named refusals, before any kernel runs — on a type with a wide kernel and on one
/// without.
#[test]
fn dot_row_cols_wide_refuses_bad_shapes() {
    for (ty, k, row_bytes) in [(GgmlType::IQ2_XXS, 256, 66), (GgmlType::Q6_K, 256, 210)] {
        let wrow = vec![0u8; row_bytes];
        let a = vec![0u8; col_bytes(ty, k)];
        let many: Vec<&[u8]> = vec![a.as_slice(); qdot::WIDE_COLS + 1];
        let mut out = vec![0.0f32; qdot::WIDE_COLS + 1];
        assert_eq!(
            qdot::dot_row_cols_wide(ty, &wrow, &many, k, &mut out),
            Err(QdotError::WideShape {
                cols: qdot::WIDE_COLS + 1,
                outs: qdot::WIDE_COLS + 1
            }),
            "{ty:?}"
        );
        assert_eq!(
            qdot::dot_row_cols_wide(ty, &wrow, &[], k, &mut []),
            Err(QdotError::WideShape { cols: 0, outs: 0 }),
            "{ty:?}"
        );
        assert_eq!(
            qdot::dot_row_cols_wide(ty, &wrow, &many[..9], k, &mut out[..10]),
            Err(QdotError::WideShape { cols: 9, outs: 10 }),
            "{ty:?}"
        );
        let short = &a[..a.len() - 1];
        assert_eq!(
            qdot::dot_row_cols_wide(ty, &wrow, &[a.as_slice(), short], k, &mut out[..2]),
            Err(QdotError::ShortActivationCol {
                have: a.len() - 1,
                need: a.len(),
                k
            }),
            "{ty:?}"
        );
        let e = qdot::dot_row_cols_wide(ty, &wrow, &many, k, &mut out)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&format!("1..={} columns", qdot::WIDE_COLS)),
            "{ty:?}: {e}"
        );
    }
}

/// IQ4_NL tile clause: random rows and the ends of [`IQ4NL_ENDS`] at each of
/// [`LEGACY_TILE_KS`] (640 is the routed down; 736 and 96 end in tail blocks),
/// then the file's first routed-expert rows — tile = `dot_row` per column, bit
/// for bit.
#[test]
#[ignore = "hw: needs the box (AVX2) and the Qwen3.8 UD-Q3_K_XL file"]
fn hw_iq4nl_tile_matches_dot_row() {
    iq_tile_synthetic(GgmlType::IQ4_NL, 18, 32, &LEGACY_TILE_KS, &IQ4NL_ENDS);
    iq_tile_real(GgmlType::IQ4_NL, 18, 32);
}

/// A synthetic Q6_K block: `ql(i)` the 128 low-nibble bytes, `qh(i)` the 64
/// high-bit bytes, `sc(j)` the 16 scales, and the f16 bits `d`; packed as
/// the scalar mirror reads a block (ggml-common.h's `block_q6_K`).
fn q6k_block(
    ql: impl Fn(usize) -> u8,
    qh: impl Fn(usize) -> u8,
    sc: impl Fn(usize) -> i8,
    d: u16,
) -> Vec<u8> {
    let mut b = vec![0u8; 210];
    for (i, x) in b[..128].iter_mut().enumerate() {
        *x = ql(i);
    }
    for (i, x) in b[128..192].iter_mut().enumerate() {
        *x = qh(i);
    }
    for (j, x) in b[192..208].iter_mut().enumerate() {
        *x = sc(j) as u8;
    }
    b[208..210].copy_from_slice(&d.to_le_bytes());
    b
}

/// The ends a Q6_K tile clause's adversarial rows are built from: every code
/// 31 (ql and qh all ones) and −32 (all zeros) under scales 127 and −128, the
/// high bits alone (ql 0, qh 0xFF: codes 16) and the low nibbles alone (ql
/// 0xFF, qh 0: codes −17) — a tile that drops `qh` or reads the scales out of
/// order moves a value — a ramp with scales that differ per sub-block, d = 0
/// and a subnormal d.
const Q6K_ENDS: [IqEnd; 9] = [
    || q6k_block(|_| 0xFF, |_| 0xFF, |_| 127, F16_ONE),
    || q6k_block(|_| 0, |_| 0, |_| 127, F16_ONE),
    || q6k_block(|_| 0, |_| 0, |_| -128, F16_NEG_ONE),
    || q6k_block(|_| 0, |_| 0xFF, |_| 127, F16_ONE),
    || q6k_block(|_| 0xFF, |_| 0, |_| -128, F16_ONE),
    || {
        q6k_block(
            |i| (i * 37) as u8,
            |i| (i * 91) as u8,
            |j| 8 * j as i8 - 60,
            0x3800,
        )
    },
    || q6k_block(|i| (i * 37) as u8, |i| (i * 91) as u8, |j| 1 + j as i8, 0),
    || q6k_block(|i| (i * 11) as u8, |i| (i * 5) as u8, |_| 127, F16_TINY),
    || {
        q6k_block(
            |i| 0x5A ^ i as u8,
            |i| 0xA5 ^ (i * 3) as u8,
            |j| (j as i8 - 8) * 15,
            0x4200,
        )
    },
];

/// Q6_K tile clause: random rows (every byte but `d`, which is a finite scale
/// of either sign) and the ends of [`Q6K_ENDS`] at k = 512 (Qwen3.8's routed
/// down), 2048 and 256 against the ±3.0 and seeded columns; then the first
/// aligned Q6_K tensor of the model file on ik's dumped column — also within
/// 1 ULP of ik's own x4 kernel, as `hw_q6k_dot_row_matches_scalar_and_predicts`
/// holds `dot_row` — the V4.1 set's `output.weight` and the Q6_K routed down
/// of a Q4_K_M Qwen3.6 file: tile = `dot_row` per column, bit for bit.
#[test]
#[ignore = "hw: needs the box (AVX2), the model file, the V4.1 shards, the Qwen3.6 Q4_K_M file and $BLOOMERY_DATA/ref"]
fn hw_q6k_tile_matches_dot_row() {
    let ty = GgmlType::Q6_K;
    let mut rng = Lcg(0x6_7113);
    for k in [512, 2048, 256] {
        let nb = k / 256;
        let row_bytes = nb * 210;
        let cols = legacy_tile_columns(ty, k, vec![]);
        let mut rows = Vec::new();
        for _ in 0..LEGACY_RANDOM_ROWS * nb {
            let mut b: Vec<u8> = (0..210).map(|_| (rng.next_u32() >> 13) as u8).collect();
            b[208..210].copy_from_slice(&rand_f16(&mut rng).to_le_bytes());
            rows.extend_from_slice(&b);
        }
        let calls = assert_tile_matches(ty, "random rows", k, row_bytes, &rows, &cols);
        eprintln!("q6k tile: random rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
        let rows = legacy_end_rows(&Q6K_ENDS, nb, |end| end());
        let calls = assert_tile_matches(ty, "end rows", k, row_bytes, &rows, &cols);
        eprintln!("q6k tile: end rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
    }

    let g = gguf::Gguf::open(model_path()).unwrap();
    let w = g
        .iter_tensors()
        .find(|w| w.ty == ty && (w.dims[0] as usize).is_multiple_of(256))
        .expect("the model must carry at least one aligned Q6_K tensor");
    let k = w.dims[0] as usize;
    let row_bytes = 210 * k / 256;
    let bytes = &g.data(w).unwrap()[..TILE_ROWS * row_bytes];
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q6k-x4-ik-dot.txt"))
        .expect("run just build-ref first (it builds and runs the x4 reference harnesses)");
    let (dump_k, ik_acol, want) = parse_ik_dot_dump(&dump);
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    assert!(want.len() >= 64, "dump needs 64 rows");
    for (r, &w) in want.iter().take(64).enumerate() {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let mut out = [f32::NAN; 2];
        qdot::dot_row_cols(ty, src, &[&ik_acol, &ik_acol], k, &mut out).unwrap();
        for got in out {
            let ulp = (got.to_bits() as i64 - w as i64).abs();
            assert!(
                ulp <= 1,
                "row {r}: tile {got:.9e} vs ik {}: {ulp} ULP apart",
                f32::from_bits(w)
            );
        }
    }
    let calls = assert_tile_matches(
        ty,
        &w.name,
        k,
        row_bytes,
        bytes,
        &coded_columns(ty, k, vec![ik_acol]),
    );
    eprintln!(
        "q6k tile: {} (k = {k}) on ik's column + seeded: {calls} calls, bit-identical, 64 rows \
         within 1 ULP of ik's x4 kernel",
        w.name
    );

    let (name, k, row_bytes, bytes) = v41_rows(ty, "output", TILE_ROWS);
    let calls = assert_tile_matches(
        ty,
        &name,
        k,
        row_bytes,
        &bytes,
        &legacy_tile_columns(ty, k, vec![]),
    );
    eprintln!("q6k tile: {name} (k = {k}): {calls} calls, c = 1..=8, bit-identical");

    let first = "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    let (name, k, row_bytes, bytes) = split_rows(first, ty, 210, 256, "ffn_down_exps", TILE_ROWS);
    let calls = assert_tile_matches(
        ty,
        &name,
        k,
        row_bytes,
        &bytes,
        &legacy_tile_columns(ty, k, vec![]),
    );
    eprintln!("q6k tile: {name} (k = {k}): {calls} calls, c = 1..=8, bit-identical");
}

/// A tile call's column count outside `1..=TILE_COLS`, an `out` of another
/// length and a short column are named refusals, before any kernel runs.
#[test]
fn dot_row_cols_refuses_bad_shapes() {
    for (ty, k, row_bytes) in [
        (GgmlType::Q3_K, 256, 110),
        (GgmlType::Q6_K, 256, 210),
        (GgmlType::Q5_1, 640, 480),
        (GgmlType::Q8_0, 640, 680),
        (GgmlType::MXFP4, 640, 340),
    ] {
        let wrow = vec![0u8; row_bytes];
        let a = vec![0u8; col_bytes(ty, k)];
        let many: Vec<&[u8]> = vec![a.as_slice(); qdot::TILE_COLS + 1];
        let mut out = vec![0.0f32; qdot::TILE_COLS + 1];
        assert_eq!(
            qdot::dot_row_cols(ty, &wrow, &many, k, &mut out),
            Err(QdotError::TileShape {
                cols: qdot::TILE_COLS + 1,
                outs: qdot::TILE_COLS + 1
            }),
            "{ty:?}"
        );
        assert_eq!(
            qdot::dot_row_cols(ty, &wrow, &[], k, &mut []),
            Err(QdotError::TileShape { cols: 0, outs: 0 }),
            "{ty:?}"
        );
        assert_eq!(
            qdot::dot_row_cols(ty, &wrow, &many[..2], k, &mut out[..3]),
            Err(QdotError::TileShape { cols: 2, outs: 3 }),
            "{ty:?}"
        );
        let short = &a[..a.len() - 1];
        assert_eq!(
            qdot::dot_row_cols(ty, &wrow, &[a.as_slice(), short], k, &mut out[..2]),
            Err(QdotError::ShortActivationCol {
                have: a.len() - 1,
                need: a.len(),
                k
            }),
            "{ty:?}"
        );
        assert_eq!(
            qdot::dot_row_cols(ty, &wrow[..row_bytes - 1], &many[..2], k, &mut out[..2]),
            Err(QdotError::ShortWeightRow {
                have: row_bytes - 1,
                need: row_bytes,
                k
            }),
            "{ty:?}"
        );
        let e = qdot::dot_row_cols(ty, &wrow, &many, k, &mut out)
            .unwrap_err()
            .to_string();
        assert!(e.contains("1..=8 columns"), "{ty:?}: {e}");
    }
}

// ------------------------------------------------------- MXFP4 tile
// Synthetic rows only: MXFP4's multi-column caller is a model file the gates do
// not read, and every block's bytes are valid codes, so random bytes and the
// end blocks cover the kernel's whole input space.

/// Row lengths of the MXFP4 tile clause: 2048 and 4096 (the routed experts'
/// widths, sixteen and thirty-two x4 groups), 736 — five groups and three q8_2
/// tail blocks — and 96, three tail blocks and no group.
const MXFP4_TILE_KS: [usize; 4] = [2048, 4096, 736, 96];

/// An MXFP4 block: the E8M0 scale `e` and the 16 code bytes from `qs`.
fn mxfp4_block(e: u8, qs: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut b = vec![e];
    b.extend((0..16).map(qs));
    b
}

/// MXFP4 end blocks: the largest code (the unsigned table's 24) and the
/// smallest (code 15, 0) at scale 1, code 0 (value 0, table 12), mixed
/// nibbles, a ramp, the two subnormal scales e = 0 and e = 1 and e = 2 (the
/// first normal one), then the largest scales e = 254 and e = 255, whose
/// products reach the f32 range's end. The first [`MXFP4_FINITE_ENDS`] stay
/// finite against any seeded column.
const MXFP4_ENDS: [IqEnd; 10] = [
    || mxfp4_block(127, |_| 0x77),
    || mxfp4_block(127, |_| 0xFF),
    || mxfp4_block(127, |_| 0x00),
    || mxfp4_block(127, |_| 0x7F),
    || mxfp4_block(130, |i| (i * 37) as u8),
    || mxfp4_block(0, |_| 0x77),
    || mxfp4_block(1, |_| 0x77),
    || mxfp4_block(2, |i| (i * 11) as u8),
    || mxfp4_block(254, |_| 0x77),
    || mxfp4_block(255, |_| 0x7F),
];
const MXFP4_FINITE_ENDS: usize = 8;

/// One call of `dot_row_cols` over each of c = 2, 3, 5 and 8 columns from the
/// start of `acols` equals the scalar mirror `dot_row_scalar` per column, bit
/// for bit: the tile against the emulator, not through the AVX2 one-column
/// kernel.
fn assert_tile_matches_emulator(
    ty: GgmlType,
    label: &str,
    k: usize,
    row_bytes: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
) {
    for r in 0..bytes.len() / row_bytes {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for c in [2, 3, 5, qdot::TILE_COLS] {
            let cols: Vec<&[u8]> = acols[..c].iter().map(Vec::as_slice).collect();
            let mut out = vec![f32::NAN; c];
            qdot::dot_row_cols(ty, src, &cols, k, &mut out).unwrap();
            for (j, (&got, a)) in out.iter().zip(&cols).enumerate() {
                let want = dot_row_scalar(ty, src, a, k).unwrap();
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{label}: row {r}, c = {c}, column {j}: tile {got:e} (bits {:#x}) vs \
                     emulator {want:e} (bits {:#x})",
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}

/// MXFP4 tile clause: at each of [`MXFP4_TILE_KS`], random rows (valid codes,
/// scales e in 112..=143) and then the ends of [`MXFP4_ENDS`], against the
/// ±3.0 columns and the seeded ones — tile = `dot_row` per column, bit for
/// bit; the random rows and the finite ends also equal the scalar mirror.
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_mxfp4_tile_matches_dot_row() {
    let ty = GgmlType::MXFP4;
    let mut rng = Lcg(0x3F_4711);
    for k in MXFP4_TILE_KS {
        let nb = k / 32;
        let row_bytes = nb * 17;
        let cols = legacy_tile_columns(ty, k, vec![]);
        let mut rows = Vec::new();
        for _ in 0..LEGACY_RANDOM_ROWS * nb {
            let e = 112 + (rng.next_u32() % 32) as u8;
            let codes: [u8; 16] = std::array::from_fn(|_| (rng.next_u32() >> 7) as u8);
            rows.extend_from_slice(&mxfp4_block(e, |i| codes[i]));
        }
        let calls = assert_tile_matches(ty, "random rows", k, row_bytes, &rows, &cols);
        assert_tile_matches_emulator(ty, "random rows", k, row_bytes, &rows, &cols);
        eprintln!("mxfp4 tile: random rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
        let rows = legacy_end_rows(&MXFP4_ENDS, nb, |end| end());
        let calls = assert_tile_matches(ty, "end rows", k, row_bytes, &rows, &cols);
        eprintln!("mxfp4 tile: end rows (k = {k}): {calls} calls, c = 1..=8, bit-identical");
        let rows = legacy_end_rows(&MXFP4_ENDS[..MXFP4_FINITE_ENDS], nb, |end| end());
        assert_tile_matches_emulator(ty, "finite end rows", k, row_bytes, &rows, &cols);
    }
}

// ------------------------------------------------- Q3_K row-lane tile
// `dot_q3k_r8_cols` and its scalar mirror over `repack_q3k_r8`'s layout
// against `dot_row` per (row, column), bit for bit, at every column count
// 1..=TILE_COLS and every tile slot a column can sit in. The bit contract
// holds by construction — the same integer sum per super-block, the same
// float step per (row, column) — so any differing value is a bug.

/// The row-lane tile's two paths, kernel first.
type R8Path = fn(&[u8], &[&[u8]], usize, &mut [[f32; qdot::Q3K_R8_ROWS]]) -> Result<(), QdotError>;
const R8_PATHS: [(&str, R8Path); 2] = [
    ("kernel", qdot::dot_q3k_r8_cols),
    ("scalar", qdot::dot_q3k_r8_cols_scalar),
];

/// Eight distinct, finite, normal f16 `d` values, none a power of two (so
/// `dcol · d` rounds): each row of a synthetic row-lane group takes its own,
/// so a permutation of the kernel's `d` lanes changes the permuted rows'
/// values.
const R8_D: [u16; 8] = [
    0x3C01, 0x3555, 0x3E66, 0x3A3D, 0x4049, 0x3801, 0x3CCD, 0x2E66,
];

/// How far past a 16-byte boundary the row-lane clause copies every group
/// and column once more: odd, so no load in the kernel may assume any
/// alignment.
const MISALIGN: usize = 7;

/// `src` copied to `MISALIGN` bytes past a 16-byte boundary: the buffer and
/// where the copy starts in it.
fn misaligned_copy(src: &[u8]) -> (Vec<u8>, usize) {
    let mut buf = vec![0u8; src.len() + 16 + MISALIGN];
    let at = (16 - buf.as_ptr().addr() % 16) % 16 + MISALIGN;
    buf[at..at + src.len()].copy_from_slice(src);
    assert_eq!(buf[at..].as_ptr().addr() % 16, MISALIGN);
    (buf, at)
}

/// Clause of one row set (a whole number of 8-row groups): repacked, then for
/// c = 1..=TILE_COLS and every starting column of the cyclic list `acols`,
/// every group's tile over the c columns from there equals `dot_row` per
/// (row, column), bit for bit — the kernel on every group, first on a copy
/// of the packed groups and of the columns at [`MISALIGN`] bytes past a
/// 16-byte boundary (so an aligned load faults on the first call), then in
/// place; the scalar mirror on the first `mirror_groups`. The outputs start
/// as a NaN canary and the call's slice is followed by one more canary
/// column, so an unwritten value and a write past the c columns both fail.
/// Returns the calls made and the time the scalar mirror took.
fn assert_r8_matches(
    label: &str,
    k: usize,
    row_bytes: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
    mirror_groups: usize,
) -> (usize, std::time::Duration) {
    let ty = GgmlType::Q3_K;
    const R8: usize = qdot::Q3K_R8_ROWS;
    assert!(
        qdot::has_tile(ty),
        "{label}: no AVX2+F16C — the kernel path would be the scalar mirror twice"
    );
    let rows = bytes.len() / row_bytes;
    assert!(
        rows.is_multiple_of(R8) && rows * row_bytes == bytes.len(),
        "{label}: {rows} rows"
    );
    let n = acols.len();
    assert!(
        n >= qdot::TILE_COLS,
        "{label}: {n} columns cannot fill a tile"
    );
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for (j, a) in acols.iter().enumerate() {
            want[r * n + j] = dot_row(ty, src, a, k).unwrap();
        }
    }
    let mut packed = vec![0u8; bytes.len()];
    qdot::repack_q3k_r8(bytes, rows, k, &mut packed).unwrap();
    let (moved, at) = misaligned_copy(&packed);
    let moved = &moved[at..at + packed.len()];
    let moved_cols: Vec<(Vec<u8>, usize)> = acols
        .iter()
        .map(|a| misaligned_copy(a.as_slice()))
        .collect();
    let moved_cols: Vec<&[u8]> = moved_cols
        .iter()
        .zip(acols)
        .map(|((b, at), a)| &b[*at..*at + a.len()])
        .collect();
    let [(_, kernel), (_, scalar)] = R8_PATHS;
    let group_bytes = R8 * row_bytes;
    let canary = f32::from_bits(0x7fc0_dead);
    let mut calls = 0;
    let mut mirror = std::time::Duration::ZERO;
    for c in 1..=qdot::TILE_COLS {
        for s in 0..n {
            let idx: Vec<usize> = (0..c).map(|i| (s + i) % n).collect();
            let cols: Vec<&[u8]> = idx.iter().map(|&j| acols[j].as_slice()).collect();
            let moved_c: Vec<&[u8]> = idx.iter().map(|&j| moved_cols[j]).collect();
            for g in 0..rows / R8 {
                let at = g * group_bytes;
                let grp = &packed[at..at + group_bytes];
                let runs = [
                    (
                        "kernel, misaligned",
                        kernel,
                        &moved[at..at + group_bytes],
                        &moved_c,
                    ),
                    ("kernel", kernel, grp, &cols),
                    ("scalar", scalar, grp, &cols),
                ];
                for (path, f, grp, cs) in runs {
                    if path == "scalar" && g >= mirror_groups {
                        continue;
                    }
                    let mut buf = vec![[canary; R8]; qdot::TILE_COLS + 1];
                    let t0 = std::time::Instant::now();
                    f(grp, cs, k, &mut buf[..c]).unwrap();
                    if path == "scalar" {
                        mirror += t0.elapsed();
                    }
                    for (slot, &j) in idx.iter().enumerate() {
                        for r in 0..R8 {
                            let (got, w) = (buf[slot][r], want[(R8 * g + r) * n + j]);
                            assert_eq!(
                                got.to_bits(),
                                w.to_bits(),
                                "{label} ({path}): group {g} row {r}, c = {c}, slot {slot} \
                                 (column {j}): tile {got:e} (bits {:#x}) vs dot_row {w:e} (bits {:#x})",
                                got.to_bits(),
                                w.to_bits()
                            );
                        }
                    }
                    for (t, extra) in buf[c..].iter().enumerate() {
                        assert!(
                            extra.iter().all(|v| v.to_bits() == canary.to_bits()),
                            "{label} ({path}): group {g}, c = {c}: output column {} past the \
                             call's columns was written: {extra:?}",
                            c + t
                        );
                    }
                    calls += 1;
                }
            }
        }
    }
    (calls, mirror)
}

/// Row-lane tile clause: two synthetic 8-row groups at the ends of the
/// folded terms — the kinds of [`q3k_ends`], and a group of u = 7 in every
/// lane, which on the all-(−128) column drives each lane's i16 sum of four
/// maddubs to its bound (8 · 7 · 128 = 7168), each row of a group with its
/// own `d` ([`R8_D`]) — against the ends' q8_K columns; then the V4.1 set's
/// routed gate stack (k = the model width; the scalar mirror on every group)
/// and V2-Lite's `ffn_gate`-1 rows on the oracle's six `ffn_norm-1` tokens —
/// tile = `dot_row` per (row, column), bit for bit, on the packed bytes and
/// on a misaligned copy ([`assert_r8_matches`]).
#[test]
#[ignore = "hw: needs the box, the V4.1 shard, the model file and $BLOOMERY_DATA/ref"]
fn hw_q3k_r8_tile_matches_dot_row() {
    let ty = GgmlType::Q3_K;
    // Rows of two super-blocks; in each, the eight rows of a group carry
    // eight different d.
    let k = 512;
    let (kinds, ends) = q3k_ends(k);
    let bound: [Q3kKind; 4] = [
        (|_| 7, |_| -32),
        (|_| 7, |_| 31),
        (|_| 7, |j| if j % 2 == 0 { -32 } else { 31 }),
        (|_| 7, |j| if j % 2 == 0 { 31 } else { -32 }),
    ];
    let mut rows = Vec::new();
    for (r, d) in R8_D.into_iter().enumerate() {
        let (u0, s0) = kinds[r % 7];
        let (u1, s1) = kinds[(r + 3) % 7];
        rows.extend_from_slice(&q3k_block(u0, s0, d));
        rows.extend_from_slice(&q3k_block(u1, s1, R8_D[(r + 3) % 8]));
    }
    for r in 0..8 {
        let (u0, s0) = bound[r % 4];
        let (u1, s1) = bound[(r + 1) % 4];
        rows.extend_from_slice(&q3k_block(u0, s0, R8_D[(r + 5) % 8]));
        rows.extend_from_slice(&q3k_block(u1, s1, R8_D[(r + 1) % 8]));
    }
    let (calls, _) = assert_r8_matches(
        "synthetic ends",
        k,
        k / 256 * 110,
        &rows,
        &coded_columns(ty, k, ends),
        2,
    );
    eprintln!(
        "q3k r8 tile: synthetic ends (k = {k}): {calls} group calls, c = 1..=8, kernel and mirror bit-identical"
    );

    let (name, k, row_bytes, bytes) = v41_rows(ty, "ffn_gate_exps", TILE_ROWS);
    let groups = TILE_ROWS / qdot::Q3K_R8_ROWS;
    let (calls, mirror) = assert_r8_matches(
        &name,
        k,
        row_bytes,
        &bytes,
        &coded_columns(ty, k, vec![]),
        groups,
    );
    eprintln!(
        "q3k r8 tile: {name} (k = {k}, {} rows): {calls} group calls, c = 1..=8, bit-identical; \
         scalar mirror on all {groups} groups: {mirror:.2?}",
        bytes.len() / row_bytes
    );

    let g = gguf::Gguf::open(model_path()).unwrap();
    let t = g
        .find("blk.1.ffn_gate_exps.weight")
        .or_else(|| g.find("blk.0.ffn_gate.weight"));
    let t = t.expect("V2-Lite carries a Q3_K ffn_gate");
    assert_eq!(t.ty, ty, "{}: type", t.name);
    let k2 = t.dims[0] as usize;
    let rb2 = k2 / 256 * 110;
    let rows2 = &g.data(t).unwrap()[..TILE_ROWS * rb2];
    let toks = oracle_f32("ffn_norm-1", k2 * 6);
    let real: Vec<Vec<u8>> = toks
        .chunks_exact(k2)
        .map(|x| {
            let mut a = vec![0u8; col_bytes(ty, k2)];
            quantize_col(ty, x, &mut a);
            a
        })
        .collect();
    let (calls, _) = assert_r8_matches(&t.name, k2, rb2, rows2, &coded_columns(ty, k2, real), 2);
    eprintln!(
        "q3k r8 tile: {} (k = {k2}) on six oracle tokens + seeded: {calls} group calls, bit-identical",
        t.name
    );
}

/// The row-lane repack and tile refuse, by name and with `out` left bit for
/// bit as it was: a column count outside `1..=TILE_COLS`, an `out` of another
/// length, a `k` off the 256-value grid, a group of another length, a short
/// column, a row count off the 8-row grid and repack buffers of another
/// length. `k = 0` is an empty product: `Ok` on empty buffers (the tile's
/// values 0), refused with a buffer of any other length.
#[test]
fn q3k_r8_refuses_bad_shapes() {
    let ty = GgmlType::Q3_K;
    let k = 256;
    let group = vec![0u8; 8 * 110];
    let two = vec![0u8; 2 * 880];
    let a = vec![0u8; col_bytes(ty, k)];
    let short = &a[..a.len() - 1];
    let n = qdot::TILE_COLS + 1;
    let many: Vec<&[u8]> = vec![a.as_slice(); n];
    let canary = f32::from_bits(0x7fc0_dead);
    for (path, f) in R8_PATHS {
        let mut out = vec![[canary; qdot::Q3K_R8_ROWS]; n];
        let mut refuses = |group: &[u8], cols: &[&[u8]], k: usize, outs: usize, want: QdotError| {
            assert_eq!(f(group, cols, k, &mut out[..outs]), Err(want), "{path}");
            assert!(
                out.iter()
                    .flatten()
                    .all(|v| v.to_bits() == canary.to_bits()),
                "{path}: the refusal {want:?} wrote into out"
            );
        };
        refuses(
            &group,
            &many,
            k,
            n,
            QdotError::TileShape { cols: n, outs: n },
        );
        refuses(&group, &[], k, 0, QdotError::TileShape { cols: 0, outs: 0 });
        refuses(
            &group,
            &many[..2],
            k,
            3,
            QdotError::TileShape { cols: 2, outs: 3 },
        );
        refuses(
            &group,
            &many[..1],
            100,
            1,
            QdotError::UnalignedK { k: 100, gran: 256 },
        );
        refuses(
            &group[..879],
            &many[..1],
            k,
            1,
            QdotError::RowGroupBytes {
                buf: "tile group",
                have: 879,
                need: 880,
            },
        );
        refuses(
            &two,
            &many[..1],
            k,
            1,
            QdotError::RowGroupBytes {
                buf: "tile group",
                have: 1760,
                need: 880,
            },
        );
        refuses(
            &group,
            &many[..1],
            0,
            1,
            QdotError::RowGroupBytes {
                buf: "tile group",
                have: 880,
                need: 0,
            },
        );
        refuses(
            &group,
            &[a.as_slice(), short],
            k,
            2,
            QdotError::ShortActivationCol {
                have: a.len() - 1,
                need: a.len(),
                k,
            },
        );
        let mut zero = [[canary; qdot::Q3K_R8_ROWS]];
        assert_eq!(f(&[], &many[..1], 0, &mut zero), Ok(()), "{path}: k = 0");
        assert!(
            zero[0].iter().all(|v| v.to_bits() == 0),
            "{path}: k = 0 gave {zero:?}"
        );
    }
    let mut dst = vec![0xA5u8; 8 * 110];
    let mut refuses = |rows: &[u8], n_rows: usize, k: usize, len: usize, want: QdotError| {
        assert_eq!(
            qdot::repack_q3k_r8(rows, n_rows, k, &mut dst[..len]),
            Err(want)
        );
        assert!(
            dst.iter().all(|&b| b == 0xA5),
            "the repack refusal {want:?} wrote into dst"
        );
    };
    refuses(
        &group[..7 * 110],
        7,
        k,
        7 * 110,
        QdotError::RowGroup { rows: 7 },
    );
    refuses(
        &group[..8 * 110 - 1],
        8,
        k,
        8 * 110,
        QdotError::RowGroupBytes {
            buf: "repack source",
            have: 8 * 110 - 1,
            need: 8 * 110,
        },
    );
    refuses(
        &group,
        8,
        k,
        8 * 110 - 1,
        QdotError::RowGroupBytes {
            buf: "repack destination",
            have: 8 * 110 - 1,
            need: 8 * 110,
        },
    );
    refuses(
        &group,
        8,
        100,
        8 * 110,
        QdotError::UnalignedK { k: 100, gran: 256 },
    );
    refuses(
        &[],
        8,
        0,
        8 * 110,
        QdotError::RowGroupBytes {
            buf: "repack destination",
            have: 8 * 110,
            need: 0,
        },
    );
    assert_eq!(qdot::repack_q3k_r8(&[], 8, 0, &mut []), Ok(()), "k = 0");
    let e = qdot::repack_q3k_r8(&group[..7 * 110], 7, k, &mut dst[..7 * 110])
        .unwrap_err()
        .to_string();
    assert!(e.contains("groups of 8"), "{e}");
    let e = qdot::repack_q3k_r8(&group, 8, k, &mut dst[..8 * 110 - 1])
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("row-lane repack destination is 879 bytes"),
        "{e}"
    );
}

// ------------------------------------------------- Q3_K row-lane unpack
// `unpack_q3k_r8` against `repack_q3k_r8`, both ways, byte for byte. Both
// layouts use every bit once, so the pair is a bijection on each 880-byte
// group super-block: any bytes are an input, NaN and infinite `d` bytes
// included, and a dropped or moved bit shows as a differing byte.

/// Rows (8, 16, and 2,304 — one V4.1 expert's gate or up) and k (256, 512,
/// 5,120) of the unpack clauses.
const UNPACK_ROWS: [usize; 3] = [8, 16, 2304];
const UNPACK_KS: [usize; 3] = [256, 512, 5120];

/// f16 `d` bytes every clause plants over its random ones: quiet and
/// signalling NaNs, both infinities, the all-ones NaN, the smallest
/// subnormal, negative zero.
const UNPACK_D: [u16; 7] = [0x7E00, 0x7C01, 0x7C00, 0xFC00, 0xFFFF, 0x0001, 0x8000];

fn random_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| (rng.next() >> 56) as u8).collect()
}

/// Where two equal-length buffers first differ, for an assertion's message.
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// unpack(repack(x)) = x over random Q3_K rows, every row's first block
/// carrying one of [`UNPACK_D`] as its `d`, at every shape of
/// [`UNPACK_ROWS`] × [`UNPACK_KS`]. `out` starts as a canary fill.
#[test]
fn q3k_r8_unpack_inverts_repack() {
    let mut rng = Rng(0x0DD5_EED5_0000_0001);
    for n_rows in UNPACK_ROWS {
        for k in UNPACK_KS {
            let row_bytes = k / 256 * 110;
            let mut rows = random_bytes(&mut rng, n_rows * row_bytes);
            for r in 0..n_rows {
                let at = r * row_bytes + 108;
                rows[at..at + 2].copy_from_slice(&UNPACK_D[r % UNPACK_D.len()].to_le_bytes());
            }
            let mut packed = vec![0u8; rows.len()];
            qdot::repack_q3k_r8(&rows, n_rows, k, &mut packed).unwrap();
            let mut back = vec![0xA5u8; rows.len()];
            qdot::unpack_q3k_r8(&packed, n_rows, k, &mut back).unwrap();
            assert!(
                back == rows,
                "{n_rows} rows, k {k}: unpack(repack(x)) differs from x at byte {:?}",
                first_diff(&back, &rows)
            );
        }
    }
}

/// repack(unpack(y)) = y over random row-lane bytes at every shape: no bit
/// of the row-lane layout is dropped on the way to Q3_K.
#[test]
fn q3k_r8_repack_inverts_unpack() {
    let mut rng = Rng(0x0DD5_EED5_0000_0002);
    for n_rows in UNPACK_ROWS {
        for k in UNPACK_KS {
            let groups = random_bytes(&mut rng, n_rows * (k / 256) * 110);
            let mut rows = vec![0xA5u8; groups.len()];
            qdot::unpack_q3k_r8(&groups, n_rows, k, &mut rows).unwrap();
            let mut again = vec![0x5Au8; groups.len()];
            qdot::repack_q3k_r8(&rows, n_rows, k, &mut again).unwrap();
            assert!(
                again == groups,
                "{n_rows} rows, k {k}: repack(unpack(y)) differs from y at byte {:?}",
                first_diff(&again, &groups)
            );
        }
    }
}

/// The unpack refuses what the repack refuses, by name and with `out` left
/// bit for bit as it was: a row count off the 8-row grid, a `groups` or `out`
/// of another length, a `k` off the 256-value grid. `k = 0` is `Ok` on empty
/// buffers and refused with an `out` of any other length.
#[test]
fn q3k_r8_unpack_refuses_bad_shapes() {
    let k = 256;
    let group = vec![0u8; 8 * 110];
    let mut dst = vec![0xA5u8; 8 * 110];
    let mut refuses = |groups: &[u8], n_rows: usize, k: usize, len: usize, want: QdotError| {
        assert_eq!(
            qdot::unpack_q3k_r8(groups, n_rows, k, &mut dst[..len]),
            Err(want)
        );
        assert!(
            dst.iter().all(|&b| b == 0xA5),
            "the unpack refusal {want:?} wrote into dst"
        );
    };
    refuses(
        &group[..7 * 110],
        7,
        k,
        7 * 110,
        QdotError::RowGroup { rows: 7 },
    );
    refuses(
        &group[..8 * 110 - 1],
        8,
        k,
        8 * 110,
        QdotError::RowGroupBytes {
            buf: "unpack source",
            have: 8 * 110 - 1,
            need: 8 * 110,
        },
    );
    refuses(
        &group,
        8,
        k,
        8 * 110 - 1,
        QdotError::RowGroupBytes {
            buf: "unpack destination",
            have: 8 * 110 - 1,
            need: 8 * 110,
        },
    );
    refuses(
        &group,
        8,
        100,
        8 * 110,
        QdotError::UnalignedK { k: 100, gran: 256 },
    );
    refuses(
        &[],
        8,
        0,
        8 * 110,
        QdotError::RowGroupBytes {
            buf: "unpack destination",
            have: 8 * 110,
            need: 0,
        },
    );
    assert_eq!(qdot::unpack_q3k_r8(&[], 8, 0, &mut []), Ok(()), "k = 0");
    let e = qdot::unpack_q3k_r8(&group, 8, k, &mut dst[..8 * 110 - 1])
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("row-lane unpack destination is 879 bytes"),
        "{e}"
    );
}

// --------------------------------------- Q4_K/Q5_K/Q6_K row lanes
// `pack_lanes` then `dot_lanes_cols` against `dot_row` per (row, column), bit
// for bit, at every column count 1..=TILE_COLS and every slot a column can sit
// in. The bit contract holds by construction — the same integer sum per half
// sub-block, the same f32 products, the same fused adds per (row, column)
// lane and the same lane sum — so any differing value is a bug.

/// Q4_K's synthetic row sets, `(label, k, rows)`: seeded rows ([`lane_rows`])
/// of one, two and three super-blocks, two and four whole chunks, and two
/// chunks with a one-super-block tail, then all-0xFF rows ([`lane_fill`]) at
/// k = 4096. The tile clause ([`hw_q4k_tile_matches_dot_row`]) and the
/// AVX-512 clause ([`hw_q4k_avx512_matches_avx2_synthetic`]) both read them.
fn q4k_synthetic_rows() -> Vec<(String, usize, Vec<u8>)> {
    let ty = GgmlType::Q4_K;
    let mut sets: Vec<(String, usize, Vec<u8>)> = [
        (256usize, 16usize, 0x5a12_0001u64),
        (512, 16, 0x5a12_0005),
        (768, 16, 0x5a12_0006),
        (2048, 16, 0x5a12_0002),
        (2304, 16, 0x5a12_0003),
        (4096, 24, 0x5a12_0004),
    ]
    .iter()
    .map(|&(k, rows, seed)| {
        (
            format!("seeded rows (k = {k})"),
            k,
            lane_rows(ty, k, rows, seed),
        )
    })
    .collect();
    sets.push((
        "0xFF rows (k = 4096)".into(),
        4096,
        lane_fill(ty, 4096, 0xFF),
    ));
    sets
}

/// Bytes of one super-block of a row-lane type, and where its f16 scales
/// sit: Q4_K and Q5_K's `d` and `dmin` at 0 and 2, Q6_K's `d` at 208.
fn lane_block(ty: GgmlType) -> (usize, &'static [usize]) {
    match ty {
        GgmlType::Q4_K => (144, &[0, 2]),
        GgmlType::Q5_K => (176, &[0, 2]),
        GgmlType::Q6_K => (210, &[208]),
        _ => panic!("no row-lane rows for {ty:?}"),
    }
}

/// `rows` synthetic rows of `ty` (Q4_K, Q5_K or Q6_K), `k` values each: every
/// byte of every block from the generator, then its f16 scales (`d` and
/// `dmin`, or Q6_K's `d`) finite normal of either sign (exponents 2^-10 ..
/// 2^5), so the codes, the high bits, the scales and mins and the products all
/// vary.
fn lane_rows(ty: GgmlType, k: usize, rows: usize, seed: u64) -> Vec<u8> {
    let (block, scales) = lane_block(ty);
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; rows * k / 256 * block];
    for b in out.chunks_exact_mut(block) {
        for x in b.iter_mut() {
            *x = rng.next_u32() as u8;
        }
        for &at in scales {
            let sign = (rng.next_u32() & 1) as u16;
            let exp = 5 + (rng.next_u32() % 16) as u16;
            let mant = (rng.next_u32() & 0x3ff) as u16;
            b[at..at + 2].copy_from_slice(&(sign << 15 | exp << 10 | mant).to_le_bytes());
        }
    }
    out
}

/// One 8-row group with every code and scale byte `fill`, its f16 scales row
/// `r`'s [`R8_D`] (`dmin`, where the type has one, row `r + 3`'s). 0xFF puts
/// every code at its top: Q4_K 15, Q5_K 31 with every high bit set (scales
/// and mins 63) — on the all-(+127) and all-(−127) columns each lane's i16 sum
/// of four maddubs reaches 8 · 31 · 127 = 31,496 — and Q6_K 63 (scales −1),
/// where each pair of maddubs reaches 4 · 63 · 127 = 32,004. 0x00 puts every
/// Q6_K code at −32 after its offset, the one-column kernel's widest
/// sign-folded sum, 8 · 32 · 127 = 32,512; its scales are set to 127 so no
/// product is zero.
fn lane_fill(ty: GgmlType, k: usize, fill: u8) -> Vec<u8> {
    let (block, scales) = lane_block(ty);
    let nb = k / 256;
    let mut out = vec![fill; 8 * nb * block];
    for (i, b) in out.chunks_exact_mut(block).enumerate() {
        for (j, &at) in scales.iter().enumerate() {
            b[at..at + 2].copy_from_slice(&R8_D[(i / nb + 3 * j) % 8].to_le_bytes());
        }
        if ty == GgmlType::Q6_K && fill == 0 {
            b[192..208].fill(0x7F);
        }
    }
    out
}

/// Clause of one row set (a whole number of 8-row groups): every group packed
/// ([`qdot::pack_lanes`], into a buffer prefilled with 0xA5) without a next
/// group, and again with the set's next group (the first, after the last) as
/// `next` into a second 0xA5 buffer, which must hold the same bytes — the
/// prefetch arm moves no value; then for c =
/// 1..=TILE_COLS and every starting column of the cyclic list `acols`, the
/// group's lanes over the c columns from there equal `dot_row` per (row,
/// column), bit for bit — on the packed group and the columns in place, then
/// on copies of both at [`MISALIGN`] bytes past a 16-byte boundary. The
/// outputs start as a NaN canary and the call's slice is followed by one more
/// canary column, so an unwritten value and a write past the c columns both
/// fail. Returns the calls made.
fn assert_lanes_match(
    ty: GgmlType,
    label: &str,
    k: usize,
    row_bytes: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
) -> usize {
    const L: usize = qdot::LANE_ROWS;
    assert!(
        qdot::has_lanes(ty),
        "{label}: {ty:?} has no row lanes on this CPU — the clause would compare nothing"
    );
    let rows = bytes.len() / row_bytes;
    assert!(
        rows.is_multiple_of(L) && rows * row_bytes == bytes.len(),
        "{label}: {rows} rows"
    );
    let n = acols.len();
    assert!(
        n >= qdot::TILE_COLS,
        "{label}: {n} columns cannot fill a call"
    );
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for (j, a) in acols.iter().enumerate() {
            want[r * n + j] = dot_row(ty, src, a, k).unwrap();
        }
    }
    let moved_cols: Vec<(Vec<u8>, usize)> = acols
        .iter()
        .map(|a| misaligned_copy(a.as_slice()))
        .collect();
    let moved_cols: Vec<&[u8]> = moved_cols
        .iter()
        .zip(acols)
        .map(|((b, at), a)| &b[*at..*at + a.len()])
        .collect();
    let group_bytes = L * row_bytes;
    let canary = f32::from_bits(0x7fc0_dead);
    let mut packed = vec![0xA5u8; qdot::lane_pack_bytes(k)];
    let mut ahead = vec![0xA5u8; qdot::lane_pack_bytes(k)];
    let groups = rows / L;
    let mut calls = 0;
    for g in 0..groups {
        let group = &bytes[g * group_bytes..(g + 1) * group_bytes];
        let h = (g + 1) % groups;
        let next = &bytes[h * group_bytes..(h + 1) * group_bytes];
        packed.fill(0xA5);
        qdot::pack_lanes(ty, group, &[], k, &mut packed).unwrap();
        ahead.fill(0xA5);
        qdot::pack_lanes(ty, group, next, k, &mut ahead).unwrap();
        if let Some(at) = packed.iter().zip(&ahead).position(|(a, b)| a != b) {
            panic!(
                "{label}: group {g} packed with group {h} as next differs from its pack \
                 without one at byte {at}: {:#04x} vs {:#04x}",
                ahead[at], packed[at]
            );
        }
        let (moved, at) = misaligned_copy(&packed);
        let moved = &moved[at..at + packed.len()];
        for c in 1..=qdot::TILE_COLS {
            for s in 0..n {
                let idx: Vec<usize> = (0..c).map(|i| (s + i) % n).collect();
                let cols: Vec<&[u8]> = idx.iter().map(|&j| acols[j].as_slice()).collect();
                let moved_c: Vec<&[u8]> = idx.iter().map(|&j| moved_cols[j]).collect();
                for (path, grp, cs) in [
                    ("in place", packed.as_slice(), &cols),
                    ("misaligned", moved, &moved_c),
                ] {
                    let mut buf = vec![[canary; L]; qdot::TILE_COLS + 1];
                    qdot::dot_lanes_cols(ty, grp, cs, k, &mut buf[..c]).unwrap();
                    for (slot, &j) in idx.iter().enumerate() {
                        for r in 0..L {
                            let (got, w) = (buf[slot][r], want[(L * g + r) * n + j]);
                            assert_eq!(
                                got.to_bits(),
                                w.to_bits(),
                                "{label} ({path}): group {g} row {r}, c = {c}, slot {slot} \
                                 (column {j}): lanes {got:e} (bits {:#x}) vs dot_row {w:e} (bits {:#x})",
                                got.to_bits(),
                                w.to_bits()
                            );
                        }
                    }
                    for (t, extra) in buf[c..].iter().enumerate() {
                        assert!(
                            extra.iter().all(|v| v.to_bits() == canary.to_bits()),
                            "{label} ({path}): group {g}, c = {c}: output column {} past the \
                             call's columns was written: {extra:?}",
                            c + t
                        );
                    }
                    calls += 1;
                }
            }
        }
    }
    calls
}

/// The seeded columns of [`coded_columns`] for `ty` plus an all-(+3.0) and an
/// all-(−3.0) column (codes +127 and −127 throughout).
fn lane_columns(ty: GgmlType, k: usize) -> Vec<Vec<u8>> {
    let ends = [3.0f32, -3.0]
        .iter()
        .map(|&v| {
            let mut a = vec![0u8; col_bytes(ty, k)];
            quantize_col(ty, &vec![v; k], &mut a);
            a
        })
        .collect();
    coded_columns(ty, k, ends)
}

/// Row-lane clause on synthetic rows: for Q5_K and Q6_K, groups of
/// seeded rows at k = 256 (one super-block, a short chunk), 2,048 and 4,096
/// (GLM's routed down and gate/up widths: two and four whole chunks) and
/// 2,304 (two chunks and a one-super-block tail), and the 0xFF group
/// ([`lane_fill`]) at 4,096, with Q6_K's 0x00 group beside it — lanes =
/// `dot_row` per (row, column), bit for bit ([`assert_lanes_match`]).
#[test]
#[ignore = "hw: the box's CPU (AVX2); reads no model file"]
fn hw_lanes_match_dot_row_synthetic() {
    // PIN(2026-10-14): Q4_K's clause moved to hw_q4k_avx512_matches_avx2_synthetic,
    // which holds it at k = 512 and 768 too.
    for ty in [GgmlType::Q5_K, GgmlType::Q6_K] {
        let (block, _) = lane_block(ty);
        for (k, rows, seed) in [
            (256usize, 16usize, 0x1a2e_0001u64),
            (2048, 16, 0x1a2e_0002),
            (2304, 16, 0x1a2e_0003),
            (4096, 24, 0x1a2e_0004),
        ] {
            let bytes = lane_rows(ty, k, rows, seed);
            let calls = assert_lanes_match(
                ty,
                "seeded rows",
                k,
                k / 256 * block,
                &bytes,
                &lane_columns(ty, k),
            );
            eprintln!(
                "{ty:?} lanes: seeded rows (k = {k}, {rows} rows): {calls} calls, bit-identical"
            );
        }
        let k = 4096;
        let fills: &[(u8, &str)] = if ty == GgmlType::Q6_K {
            &[(0xFF, "0xFF group"), (0x00, "0x00 group")]
        } else {
            &[(0xFF, "0xFF group")]
        };
        for &(fill, label) in fills {
            let calls = assert_lanes_match(
                ty,
                label,
                k,
                k / 256 * block,
                &lane_fill(ty, k, fill),
                &lane_columns(ty, k),
            );
            eprintln!("{ty:?} lanes: {label} (k = {k}): {calls} calls, bit-identical");
        }
    }
}

/// `dot_row` on a Q4_K row refuses a `BLOOMERY_QDOT_AVX512` value other than
/// `0` or `1` by name on every architecture and CPU, before it picks a body:
/// the test reruns itself in a fresh process with the lever at `2`, since
/// [`qdot::q4k_avx512`] reads it once a process.
#[test]
fn dot_row_refuses_a_qdot_lever_value_by_name() {
    // The fresh process's own argument: a filter no test's name is.
    const FRESH: &str = "a-fresh-process-for-the-qdot-lever";
    if !std::env::args().any(|a| a == FRESH) {
        let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .args([
                "--exact",
                "dot_row_refuses_a_qdot_lever_value_by_name",
                FRESH,
            ])
            .env("BLOOMERY_QDOT_AVX512", "2")
            .output()
            .expect("the fresh process");
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("1 passed"),
            "the fresh process: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    let ty = GgmlType::Q4_K;
    let row = lane_rows(ty, 256, 1, 0x5a12_0001);
    let col = &lane_columns(ty, 256)[0];
    let dot = std::panic::catch_unwind(|| qdot::dot_row(ty, &row, col, 256));
    let e = dot.expect_err("a refused lever value must stop dot_row");
    let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(
        e.contains("BLOOMERY_QDOT_AVX512=\"2\""),
        "the refusal names the lever, got {e:?}"
    );
    println!("refusal: {e}");
}

/// Q4_K's selected one-column body equals its AVX2 body and scalar mirror
/// bit for bit on seeded and full-code rows. Its row lanes equal dot_row.
/// The printed line names the selected body.
#[test]
#[ignore = "hw: the box's CPU (AVX2; the AVX-512 VNNI bodies where it has them); reads no model file"]
#[cfg(target_arch = "x86_64")]
fn hw_q4k_avx512_matches_avx2_synthetic() {
    let ty = GgmlType::Q4_K;
    let isa = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni");
    let wanted = std::env::var_os("BLOOMERY_QDOT_AVX512").is_none_or(|v| v == "1");
    assert_eq!(
        qdot::q4k_avx512(),
        isa && wanted,
        "q4k_avx512 must pick the AVX-512 VNNI bodies exactly where the CPU has their ISA and \
         BLOOMERY_QDOT_AVX512 is not 0"
    );
    let body = if qdot::q4k_avx512() {
        "AVX-512 VNNI"
    } else {
        "AVX2 (no AVX-512 VNNI body on this CPU or under BLOOMERY_QDOT_AVX512=0)"
    };
    let (block, _) = lane_block(ty);
    for (label, k, bytes) in &q4k_synthetic_rows() {
        let (k, row_bytes) = (*k, k / 256 * block);
        let acols = lane_columns(ty, k);
        for (r, src) in bytes.chunks_exact(row_bytes).enumerate() {
            for (j, a) in acols.iter().enumerate() {
                let got = dot_row(ty, src, a, k).unwrap();
                for (side, want) in [
                    ("dot_row_avx2", dot_row_avx2(ty, src, a, k).unwrap()),
                    ("dot_row_scalar", dot_row_scalar(ty, src, a, k).unwrap()),
                ] {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{label}: row {r}, column {j}: dot_row ({body}) {got:e} (bits {:#x}) \
                         vs {side} {want:e} (bits {:#x})",
                        got.to_bits(),
                        want.to_bits()
                    );
                }
            }
        }
        // PIN(2026-10-14): tile fixtures moved to hw_q4k_tile_matches_dot_row.
        let lanes = assert_lanes_match(ty, label, k, row_bytes, bytes, &acols);
        eprintln!(
            "Q4_K {body}: {label}, {} rows x {} columns: dot_row = dot_row_avx2 = scalar, \
             {lanes} lane calls = dot_row, bit-identical",
            bytes.len() / row_bytes,
            acols.len()
        );
    }
}

/// Row-lane clause on the V4.1 set's first Q4_K and first Q5_K routed down
/// stacks and its one Q6_K tensor, `output.weight` ([`TILE_ROWS`] rows each),
/// against the seeded columns — lanes = `dot_row` per (row, column), bit for
/// bit.
#[test]
#[ignore = "hw: needs the box and the V4.1 shards"]
fn hw_lanes_match_dot_row_v41() {
    for (ty, prefer) in [
        (GgmlType::Q4_K, "ffn_down_exps"),
        (GgmlType::Q5_K, "ffn_down_exps"),
        (GgmlType::Q6_K, "output"),
    ] {
        let (name, k, row_bytes, bytes) = v41_rows(ty, prefer, TILE_ROWS);
        let calls = assert_lanes_match(ty, &name, k, row_bytes, &bytes, &lane_columns(ty, k));
        eprintln!("{ty:?} lanes: {name} (k = {k}, {TILE_ROWS} rows): {calls} calls, bit-identical");
    }
}

/// The row lanes refuse, by name and with their outputs left as they were: a
/// type without lanes, a `k` off the 256-value grid, a pack's rows, next rows
/// or group of another length, a column count outside `1..=TILE_COLS`, an `out` of
/// another length, a group of another length and a short column. `k = 0`
/// packs nothing.
#[test]
#[ignore = "hw: the box's CPU (a type has row lanes only with AVX2+FMA+F16C)"]
fn hw_lanes_refuse_bad_shapes() {
    let ty = GgmlType::Q4_K;
    assert!(
        qdot::has_lanes(ty),
        "{ty:?} has no row lanes on this CPU: every shape would be refused as NoLanes"
    );
    let k = 256;
    let rows = vec![0u8; 8 * 144];
    let need = qdot::lane_pack_bytes(k);
    assert_eq!(need, 2560, "one packed super-block");
    let a = vec![0u8; col_bytes(ty, k)];
    let short = &a[..a.len() - 1];
    let canary_b = 0x5Au8;
    let mut packed = vec![canary_b; need + 1];
    let mut pack = |w: GgmlType, rows: &[u8], k: usize, len: usize, want: QdotError| {
        assert_eq!(
            qdot::pack_lanes(w, rows, &[], k, &mut packed[..len]),
            Err(want)
        );
        assert!(
            packed.iter().all(|&b| b == canary_b),
            "the refusal {want:?} wrote into packed"
        );
    };
    pack(
        GgmlType::Q5_0,
        &rows,
        k,
        need,
        QdotError::NoLanes(GgmlType::Q5_0),
    );
    pack(
        GgmlType::Q3_K,
        &rows,
        k,
        need,
        QdotError::NoLanes(GgmlType::Q3_K),
    );
    pack(
        ty,
        &rows,
        100,
        need,
        QdotError::UnalignedK { k: 100, gran: 256 },
    );
    pack(
        ty,
        &rows[..rows.len() - 1],
        k,
        need,
        QdotError::RowGroupBytes {
            buf: "lane rows",
            have: 8 * 144 - 1,
            need: 8 * 144,
        },
    );
    pack(
        GgmlType::Q5_K,
        &rows,
        k,
        need,
        QdotError::RowGroupBytes {
            buf: "lane rows",
            have: 8 * 144,
            need: 8 * 176,
        },
    );
    pack(
        GgmlType::Q6_K,
        &rows,
        k,
        need,
        QdotError::RowGroupBytes {
            buf: "lane rows",
            have: 8 * 144,
            need: 8 * 210,
        },
    );
    pack(
        ty,
        &rows,
        k,
        need + 1,
        QdotError::RowGroupBytes {
            buf: "lane group",
            have: need + 1,
            need,
        },
    );
    for next in [
        &rows[..rows.len() - 1],
        &[rows.as_slice(), &[0]].concat()[..],
    ] {
        assert_eq!(
            qdot::pack_lanes(ty, &rows, next, k, &mut packed[..need]),
            Err(QdotError::RowGroupBytes {
                buf: "lane next rows",
                have: next.len(),
                need: 8 * 144,
            })
        );
        assert!(
            packed.iter().all(|&b| b == canary_b),
            "the next-rows refusal at {} bytes wrote into packed",
            next.len()
        );
    }
    assert_eq!(qdot::pack_lanes(ty, &[], &[], 0, &mut []), Ok(()), "k = 0");

    let group = vec![0u8; need];
    let many: Vec<&[u8]> = vec![a.as_slice(); qdot::TILE_COLS + 1];
    let canary = f32::from_bits(0x7fc0_dead);
    let mut out = vec![[canary; qdot::LANE_ROWS]; qdot::TILE_COLS + 1];
    let mut dot =
        |w: GgmlType, group: &[u8], cols: &[&[u8]], k: usize, outs: usize, want: QdotError| {
            assert_eq!(
                qdot::dot_lanes_cols(w, group, cols, k, &mut out[..outs]),
                Err(want)
            );
            assert!(
                out.iter()
                    .flatten()
                    .all(|v| v.to_bits() == canary.to_bits()),
                "the refusal {want:?} wrote into out"
            );
        };
    dot(
        GgmlType::Q5_0,
        &group,
        &many[..1],
        k,
        1,
        QdotError::NoLanes(GgmlType::Q5_0),
    );
    dot(
        ty,
        &group,
        &[],
        k,
        0,
        QdotError::TileShape { cols: 0, outs: 0 },
    );
    dot(
        ty,
        &group,
        &many,
        k,
        qdot::TILE_COLS + 1,
        QdotError::TileShape {
            cols: qdot::TILE_COLS + 1,
            outs: qdot::TILE_COLS + 1,
        },
    );
    dot(
        ty,
        &group,
        &many[..2],
        k,
        1,
        QdotError::TileShape { cols: 2, outs: 1 },
    );
    dot(
        ty,
        &group,
        &many[..1],
        100,
        1,
        QdotError::UnalignedK { k: 100, gran: 256 },
    );
    dot(
        ty,
        &group[..need - 1],
        &many[..1],
        k,
        1,
        QdotError::RowGroupBytes {
            buf: "lane group",
            have: need - 1,
            need,
        },
    );
    dot(
        ty,
        &group,
        &[a.as_slice(), short],
        k,
        2,
        QdotError::ShortActivationCol {
            have: short.len(),
            need: a.len(),
            k,
        },
    );
    let e = QdotError::NoLanes(GgmlType::Q5_0).to_string();
    assert!(
        e.contains("no row-lane qdot kernel for weight type Q5_0"),
        "{e}"
    );
}

// ------------------------------------------------ V4.1 card-rule kernels
//
// The pure gates of the card-rule kernels (q8_1, the Q3_K/Q4_K dots, the
// SwiGLU): AVX2 against the scalar mirror bit for bit on random and edge
// inputs, and every m-column call against its columns' one-column calls.
// The bits the card itself produces are the box gate's to check
// (gate_deepseek41_moe's `cardrule` site).

/// The deterministic generator the card-rule gates draw from.
struct CardRng(u64);

impl CardRng {
    fn new(seed: u64) -> CardRng {
        CardRng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A finite f32 in `[-span, span]`, every 61st scaled up so a block's
    /// amax is not always near the span.
    fn value(&mut self, span: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0;
        if self.next_u64().is_multiple_of(61) {
            u * 8.0 * span
        } else {
            u * span
        }
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

/// A random finite f16 bit pattern with a small exponent (a super-block
/// scale a real row carries).
fn small_f16(rng: &mut CardRng) -> u16 {
    let e = 12 + rng.next_u64() % 8; // 2^-15 .. 2^-8 times the mantissa
    ((e as u16) << 10) | (rng.next_u64() as u16 & 0x3ff)
}

/// A random Q3_K weight row of `n_sb` super-blocks with finite scales.
fn card_q3k_row(rng: &mut CardRng, n_sb: usize) -> Vec<u8> {
    let mut row = vec![0u8; 110 * n_sb];
    for sb in &mut row.chunks_mut(110) {
        for b in sb.iter_mut() {
            *b = rng.byte();
        }
        let d = small_f16(rng);
        sb[108] = d as u8;
        sb[109] = (d >> 8) as u8;
    }
    row
}

/// A random Q4_K weight row of `n_sb` super-blocks with finite scales.
fn card_q4k_row(rng: &mut CardRng, n_sb: usize) -> Vec<u8> {
    let mut row = vec![0u8; 144 * n_sb];
    for sb in &mut row.chunks_mut(144) {
        for b in sb.iter_mut() {
            *b = rng.byte();
        }
        let (d, dmin) = (small_f16(rng), small_f16(rng));
        sb[0] = d as u8;
        sb[1] = (d >> 8) as u8;
        sb[2] = dmin as u8;
        sb[3] = (dmin >> 8) as u8;
    }
    row
}

/// A random activation column of `n_sb` super-blocks (256 values each).
fn card_act_col(rng: &mut CardRng, n_sb: usize) -> Vec<f32> {
    (0..256 * n_sb).map(|_| rng.value(2.0)).collect()
}

fn card_bits_named(a: f32, b: f32, what: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "card kernel bits differ ({what}): {a:?} ({:#x}) vs {b:?} ({:#x})",
        a.to_bits(),
        b.to_bits()
    );
}

/// One column's q8_1 forms by both bodies, held equal field by field.
fn card_q8_1_both(col: &[f32], what: &str) -> qdot::CardQ81 {
    let (avx, scal) = (qdot::card_q8_1(col), qdot::card_q8_1_scalar(col));
    let diff: Vec<String> = avx
        .codes()
        .iter()
        .zip(scal.codes())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(12)
        .map(|(i, (a, b))| format!("{i}: {a}!={b} (x={:?})", col[i]))
        .collect();
    assert!(diff.is_empty(), "{what}: codes differ: {}", diff.join(", "));
    assert_eq!(avx.s8(), scal.s8(), "{what}: sums differ");
    assert_eq!(avx.refused(), scal.refused(), "{what}: refusals differ");
    for (b, (a, s)) in avx.d8().iter().zip(scal.d8()).enumerate() {
        assert_eq!(
            a.to_bits(),
            s.to_bits(),
            "{what}: block {b}'s scale differs"
        );
    }
    avx
}

#[test]
#[ignore = "hw: needs the box's AVX2 (card_q8_1's vector body)"]
fn hw_card_q8_1_scalar_matches_avx2_and_edges() {
    let mut rng = CardRng::new(0xC0DE_41A1);
    // Ties: every block holds 127 (scale exactly 1) and the values ±(j +
    // 0.5), ±j and ±(j + 0.25) for j = 0..42 — a half-even rounding moves
    // every even j's tie down by one.
    let mut ties = vec![0.0f32; 512];
    for (i, v) in ties.iter_mut().enumerate() {
        let j = i % 128;
        let sign = if (i / 128) % 2 == 0 { 1.0 } else { -1.0 };
        *v = match j {
            0 => 127.0,
            _ => {
                let m = ((j - 1) / 3) as f32;
                sign * [m + 0.5, m, m + 0.25][(j - 1) % 3]
            }
        };
    }
    let q = card_q8_1_both(&ties, "ties");
    for (i, (&v, &c)) in ties.iter().zip(q.codes()).enumerate() {
        let want = (v.abs() + 0.5).floor().min(127.0) * v.signum();
        assert_eq!(c as f32, want, "value {i} = {v} rounds half away from zero");
    }
    // Edges around ties at every code magnitude (scale 1), the clamp, and
    // the near-tie neighbours a half-even or truncating body misplaces.
    let mut near = vec![0.0f32; 1024];
    near[0] = 127.0;
    let mut i = 1;
    for m in 0..=126i32 {
        for frac in [0.5f32, 0.49999997, 0.50000006] {
            if i + 1 < near.len() {
                near[i] = m as f32 + frac;
                near[i + 1] = -(m as f32 + frac);
                i += 2;
            }
        }
    }
    for b in 1..near.len() / 128 {
        near[128 * b] = 127.0;
    }
    card_q8_1_both(&near, "near ties");
    // An all-zero block (scale 1, codes 0), a block whose scale is
    // subnormal, and one whose scale underflows to 0: |v|/0 is inf (code
    // ±127) and 0/0 NaN (code 0), as the card's saturating convert gives.
    let mut tiny = vec![0.0f32; 512];
    for (j, v) in tiny[128..256].iter_mut().enumerate() {
        *v = (j as f32 - 64.0) * 1e-40;
    }
    tiny[256] = f32::from_bits(1);
    tiny[257] = -f32::from_bits(1);
    tiny[384] = f32::MIN_POSITIVE;
    tiny[385] = -f32::MIN_POSITIVE / 3.0;
    let q = card_q8_1_both(&tiny, "tiny");
    assert_eq!(q.d8()[0].to_bits(), 1.0f32.to_bits());
    assert!(q.codes()[..128].iter().all(|&c| c == 0));
    assert_eq!(
        q.d8()[2].to_bits(),
        0.0f32.to_bits(),
        "the underflowed scale"
    );
    assert_eq!(
        (q.codes()[256], q.codes()[257], q.codes()[258]),
        (127, -127, 0)
    );
    // Random columns of every shape the walks take, with an all-zero block
    // and a non-finite block (refused: NaN scale, zero codes and sums).
    for n_sb in [1usize, 2, 3, 9, 16] {
        for case in 0..8 {
            let mut col = card_act_col(&mut rng, n_sb);
            match case % 4 {
                1 => col[..128].fill(0.0),
                2 => col[129] = f32::NAN,
                3 => *col.last_mut().unwrap() = f32::NEG_INFINITY,
                _ => {}
            }
            let q = card_q8_1_both(&col, &format!("random n_sb={n_sb} case={case}"));
            for (b, &bad) in q.refused().iter().enumerate() {
                let blk = &col[128 * b..128 * b + 128];
                assert_eq!(
                    bad,
                    blk.iter().any(|v| !v.is_finite()),
                    "block {b}'s refusal"
                );
                if bad {
                    assert!(q.d8()[b].is_nan());
                    assert!(q.codes()[128 * b..128 * b + 128].iter().all(|&c| c == 0));
                    assert!(q.s8()[4 * b..4 * b + 4].iter().all(|&s| s == 0));
                }
            }
            for g in 0..q.s8().len() {
                let sum: i32 = q.codes()[32 * g..32 * g + 32]
                    .iter()
                    .map(|&c| c as i32)
                    .sum();
                assert_eq!(q.s8()[g], sum, "group {g}'s sum");
            }
        }
    }
}

/// The f64 dot of a dequantized weight row against a card-q8_1 column's
/// decoded values (`code · d8`), and the sum of the terms' magnitudes.
fn card_dot_f64(ty: GgmlType, row: &[u8], col: &qdot::CardQ81) -> (f64, f64) {
    let mut w = vec![0.0f32; col.k()];
    dequant_row(ty, row, &mut w).unwrap();
    let (mut dot, mut mag) = (0.0f64, 0.0f64);
    for (j, &wj) in w.iter().enumerate() {
        let a = col.codes()[j] as f64 * col.d8()[j / 128] as f64;
        dot += wj as f64 * a;
        mag += (wj as f64 * a).abs();
    }
    (dot, mag)
}

#[test]
#[ignore = "hw: needs the box's AVX2 (the card dots' vector bodies)"]
fn hw_card_dots_scalar_match_avx2_and_columns_match_one() {
    let mut rng = CardRng::new(0xC0DE_41A2);
    // Odd super-block counts run the Q3_K walk's guarded tail, counts off a
    // multiple of four the Q4_K walk's, and 8, 9, 16 its split prefix.
    for n_sb in [1usize, 2, 3, 5, 8, 9, 16] {
        let q3k = card_q3k_row(&mut rng, n_sb);
        let q4k = card_q4k_row(&mut rng, n_sb);
        for m in [1usize, 2, 3, 8, 9, 16] {
            let cols: Vec<qdot::CardQ81> = (0..m)
                .map(|_| qdot::card_q8_1(&card_act_col(&mut rng, n_sb)))
                .collect();
            let refs: Vec<&qdot::CardQ81> = cols.iter().collect();
            for (ty, row) in [(GgmlType::Q3_K, &q3k), (GgmlType::Q4_K, &q4k)] {
                let mut avx = vec![0.0f32; m];
                let mut scal = vec![0.0f32; m];
                if ty == GgmlType::Q3_K {
                    qdot::card_q3k_dot_row_cols(row, &refs, &mut avx).unwrap();
                    qdot::card_q3k_dot_row_cols_scalar(row, &refs, &mut scal).unwrap();
                } else {
                    qdot::card_q4k_dot_row_cols(row, &refs, &mut avx).unwrap();
                    qdot::card_q4k_dot_row_cols_scalar(row, &refs, &mut scal).unwrap();
                }
                for c in 0..m {
                    let one = if ty == GgmlType::Q3_K {
                        qdot::card_q3k_dot_row(row, &cols[c]).unwrap()
                    } else {
                        qdot::card_q4k_dot_row(row, &cols[c]).unwrap()
                    };
                    let at = format!("{ty:?} n_sb={n_sb} m={m} c={c}");
                    card_bits_named(avx[c], one, &format!("{at}: m-column vs one-column"));
                    card_bits_named(avx[c], scal[c], &format!("{at}: AVX2 vs scalar"));
                    // The walk is a dot of this row: the f64 dot of its
                    // dequantized values within f32 rounding of the terms.
                    let (want, mag) = card_dot_f64(ty, row, &cols[c]);
                    assert!(
                        (avx[c] as f64 - want).abs() <= 1e-5 * mag.max(f64::MIN_POSITIVE),
                        "{at}: {} against the f64 dot {want} (terms' magnitude {mag})",
                        avx[c]
                    );
                }
            }
        }
        // A refused activation block gives a NaN dot on both bodies.
        let mut bad = card_act_col(&mut rng, n_sb);
        bad[100] = f32::NAN;
        let col = qdot::card_q8_1(&bad);
        assert!(col.refused_any());
        for v in [
            qdot::card_q3k_dot_row(&q3k, &col).unwrap(),
            qdot::card_q3k_dot_row_scalar(&q3k, &col).unwrap(),
            qdot::card_q4k_dot_row(&q4k, &col).unwrap(),
            qdot::card_q4k_dot_row_scalar(&q4k, &col).unwrap(),
        ] {
            assert!(v.is_nan(), "a refused block's dot is NaN");
        }
    }
}

#[test]
fn card_dots_reject_bad_shapes() {
    let mut rng = CardRng::new(7);
    let col = qdot::card_q8_1(&card_act_col(&mut rng, 2));
    let q3k = card_q3k_row(&mut rng, 2);
    // A short row, and a Q3_K row handed to the Q4_K walk, are named errors
    // on every entry point.
    for r in [
        qdot::card_q3k_dot_row(&q3k[..109 * 2], &col),
        qdot::card_q3k_dot_row_scalar(&q3k[..109 * 2], &col),
        qdot::card_q4k_dot_row(&q3k, &col),
        qdot::card_q4k_dot_row_scalar(&q3k, &col),
    ] {
        assert!(matches!(r, Err(QdotError::ShortWeightRow { .. })), "{r:?}");
    }
    // An output slice that does not hold one value per column, and no
    // columns at all.
    let other = qdot::card_q8_1(&card_act_col(&mut rng, 2));
    let mut out = [0.0f32; 2];
    let r = qdot::card_q3k_dot_row_cols(&q3k, &[&other, &col], &mut out[..1]);
    assert!(
        matches!(r, Err(QdotError::TileShape { cols: 2, outs: 1 })),
        "{r:?}"
    );
    let r = qdot::card_q4k_dot_row_cols_scalar(&q3k, &[], &mut out[..0]);
    assert!(
        matches!(r, Err(QdotError::TileShape { cols: 0, outs: 0 })),
        "{r:?}"
    );
    // Columns of two lengths.
    let short = qdot::card_q8_1(&card_act_col(&mut rng, 1));
    let r = qdot::card_q3k_dot_row_cols(&q3k, &[&col, &short], &mut out);
    assert!(
        matches!(r, Err(QdotError::ShortActivationCol { .. })),
        "{r:?}"
    );
    // A column that is not whole super-blocks never becomes a form.
    let r = std::panic::catch_unwind(|| qdot::card_q8_1(&[0.5f32; 128]));
    assert!(r.is_err());
}

#[test]
fn expf_ik_scalar_tracks_libm() {
    // A sanity net for the port (the bit-exact claim is the box sweep
    // below): within 2 ulp of libm over silu's whole input range.
    let mut x = -104.0f32;
    while x < 104.0 {
        let got = qdot::expf_ik_scalar(x);
        let want = x.exp();
        let both_inf = got.is_infinite() && want.is_infinite();
        let ulp = (want.abs().max(f32::MIN_POSITIVE)) * 2.0f32.powi(-23);
        assert!(
            both_inf || (got - want).abs() <= 2.0 * ulp,
            "expf_ik({x}) = {got} against libm {want}"
        );
        x += 0.03125;
    }
    // The escape paths: overflow to inf past 192*ln2, underflow to 0 far
    // below, and exp(0) = 1 exactly.
    assert_eq!(qdot::expf_ik_scalar(0.0).to_bits(), 1.0f32.to_bits());
    assert!(qdot::expf_ik_scalar(200.0).is_infinite());
    assert_eq!(qdot::expf_ik_scalar(-200.0).to_bits(), 0.0f32.to_bits());
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box's AVX2 (card_expf's vector body, v_expf)"]
fn hw_v_expf_equals_expf_ik_over_f32_range() {
    // Every f32 whose exponent field is 112..=143 — all of [2^-15, 2^17) in
    // magnitude, both signs, 2^29 values: the polynomial path (|n| <= 126),
    // both escape steps and the overflow to inf all inside — and every
    // 4099th pattern of the rest, zeros, subnormals, infinities and NaNs
    // included. The vector side is `card_expf` (the SwiGLU's `v_expf`), the
    // scalar `expf_ik_scalar` (the device's `expf_ik`); a NaN input must give
    // a NaN on both, every other input the same bits.
    const BATCH: usize = 1 << 16;
    let mut xs = Vec::with_capacity(BATCH);
    let mut out = vec![0.0f32; BATCH];
    let mut checked = 0u64;
    let mut run = |xs: &mut Vec<f32>, out: &mut [f32]| {
        qdot::card_expf(xs, &mut out[..xs.len()]);
        for (&x, &got) in xs.iter().zip(out.iter()) {
            let want = qdot::expf_ik_scalar(x);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "v_expf({x:e} = {:#010x}) = {got:e} ({:#010x}) against expf_ik {want:e} ({:#010x})",
                x.to_bits(),
                got.to_bits(),
                want.to_bits()
            );
        }
        checked += xs.len() as u64;
        xs.clear();
    };
    let dense = (112u32 << 23)..(144u32 << 23);
    let sparse = (0..(1u32 << 31))
        .step_by(4099)
        .filter(|b| !dense.contains(b))
        .chain([0x7f80_0000, 0x7f80_0001, 0x7fc0_0000]);
    for sign in [0u32, 0x8000_0000] {
        for bits in dense.clone().chain(sparse.clone()) {
            xs.push(f32::from_bits(sign | bits));
            if xs.len() == BATCH {
                run(&mut xs, &mut out);
            }
        }
    }
    run(&mut xs, &mut out);
    assert!(checked > 1 << 29, "the sweep covered {checked} inputs");
}

#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box's AVX2 (card_swiglu_clamp's vector body)"]
fn hw_card_swiglu_clamp_scalar_matches_avx2() {
    let mut rng = CardRng::new(0xC0DE_41A3);
    let mut gate = vec![0.0f32; 257];
    let mut up = vec![0.0f32; 257];
    for i in 0..257 {
        gate[i] = rng.value(30.0);
        up[i] = rng.value(30.0);
    }
    // Clamp crossings both ways, the clamp off, the escape paths (|g| past
    // 87), and NaN operands (a NaN silu passes its clamp, a NaN up becomes
    // limit).
    gate[0] = 20.0;
    up[0] = 20.0;
    gate[1] = -20.0;
    up[1] = -20.0;
    gate[2] = f32::NAN;
    up[3] = f32::NAN;
    gate[4] = 95.0;
    gate[5] = -95.0;
    gate[6] = 150.0;
    gate[7] = -150.0;
    let mut avx = vec![0.0f32; 257];
    let mut scal = vec![0.0f32; 257];
    for limit in [0.0f32, 5.0, 11.5, 1e-6] {
        qdot::card_swiglu_clamp(&gate, &up, limit, &mut avx);
        for i in 0..257 {
            scal[i] = qdot::card_swiglu_clamp_1(gate[i], up[i], limit);
        }
        for i in 0..257 {
            assert!(
                avx[i].to_bits() == scal[i].to_bits() || (avx[i].is_nan() && scal[i].is_nan()),
                "swiglu({}, {}, {limit}) = {} against {}",
                gate[i],
                up[i],
                avx[i],
                scal[i]
            );
        }
    }
}

// ------------------------------------------------------- NVFP4 blocks x Q8_2_X4
// NVFP4's dot has no ik source and no model-file caller here, so its clauses are synthetic: random blocks of valid
// scales and every code nibble, held to the exact f64 dot of the decoded weights and the column's values within a
// derived band, to the scalar mirror bit for bit, and to the scale table that decodes its bytes.
//
// The band, derived before the run. Every integer sum is exact (a lane partial is at most 4·12·127 and a group sum
// 16·12·127, far under 2^24) and so is each scale times an activation scale (a 4-bit and an 8-bit significand), so the
// only roundings are the FMAs that fold a term into its lane (`nb / 2` groups, and two tail blocks for an odd `nb`,
// each one rounding of the product and the sum together) and the eight-lane hsum's three levels: `n = nb / 2 +
// 2·(nb % 2) + 3`, and `|dot − exact| ≤ γ_n · Σ|w·x|`, since a lane's terms sum in absolute value to at most the
// absolute products of its own values. The f64 reference's own sum adds `k · 2^-53 · Σ|w·x|`.

/// Row lengths of the NVFP4 clauses: 64 (a lone tail block, two q8_2 tail blocks and no group), 128 (one x4 group), 192
/// (a group and a tail), 576 (four groups and a tail), 2048 and 4096 (sixteen and thirty-two groups).
const NVFP4_BLOCK_KS: [usize; 6] = [64, 128, 192, 576, 2048, 4096];

/// One NVFP4 block: the four scale bytes, then the 32 code bytes `qs(i)` gives.
fn nvfp4_block(d: [u8; 4], qs: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut b = d.to_vec();
    b.extend((0..32).map(qs));
    b
}

/// A finite UE4M3 scale byte: in a window where every group of a row matters (exponents 6 to 9), or anywhere from zero
/// and the subnormals to the largest, 0x7E.
fn nvfp4_scale(rng: &mut Lcg, window: bool) -> u8 {
    if window {
        0x30 + (rng.next_u32() % 0x20) as u8
    } else {
        (rng.next_u32() % 0x7F) as u8
    }
}

/// `rows` rows of `nb` random blocks: any code nibble (negative zero and both extremes among them), the scales of
/// alternate blocks from the window and from the whole range.
fn nvfp4_block_rows(rng: &mut Lcg, rows: usize, nb: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for r in 0..rows * nb {
        let d = std::array::from_fn(|_| nvfp4_scale(rng, r % 2 == 0));
        let codes: [u8; 32] = std::array::from_fn(|_| (rng.next_u32() >> 7) as u8);
        out.extend(nvfp4_block(d, |i| codes[i]));
    }
    out
}

/// Rows of `nb` equal blocks at the ends of the input space: the largest scale (0x7E) with every code +12 and every code
/// -12, the smallest subnormal scale, a zero scale, and every code negative zero.
fn nvfp4_block_end_rows(nb: usize) -> Vec<Vec<u8>> {
    [
        ([0x7E; 4], 0x77u8),
        ([0x7E; 4], 0xFF),
        ([0x01; 4], 0x77),
        ([0x00; 4], 0x77),
        ([0x38; 4], 0x88),
        ([0x7E, 0x00, 0x01, 0x38], 0x5A),
    ]
    .into_iter()
    .map(|(d, code)| (0..nb).flat_map(|_| nvfp4_block(d, |_| code)).collect())
    .collect()
}

/// Columns for a row of `k` values, each through `quantize_col`: seeded activations of four magnitudes, one with a lone
/// outlier a block, the constants ±3.0 and zeros — [`qdot::TILE_COLS`] of them.
fn nvfp4_block_columns(rng: &mut Lcg, k: usize) -> Vec<Vec<u8>> {
    let mut xs: Vec<Vec<f32>> = [1.0f32, 40.0, 0.02, 7.5]
        .into_iter()
        .map(|amp| (0..k).map(|_| amp * rng.unit()).collect())
        .collect();
    xs.push(
        (0..k)
            .map(|i| if i % 32 == 7 { 90.0 } else { 0.25 * rng.unit() })
            .collect(),
    );
    xs.extend([vec![3.0; k], vec![-3.0; k], vec![0.0; k]]);
    xs.into_iter()
        .map(|x| {
            let mut col = vec![0u8; col_bytes(GgmlType::NVFP4, k)];
            quantize_col(GgmlType::NVFP4, &x, &mut col);
            col
        })
        .collect()
}

/// `(Σ w·x, Σ |w·x|)` in f64 of a row's decoded weights (`dequant_row`, exact in f32) and a column's values.
fn nvfp4_block_exact(row: &[u8], col: &[u8], k: usize) -> (f64, f64) {
    let mut w = vec![0.0f32; k];
    dequant_row(GgmlType::NVFP4, row, &mut w).unwrap();
    w.iter()
        .zip(restore_q82x4(col))
        .fold((0.0, 0.0), |(v, va), (&w, x)| {
            let p = f64::from(w) * x;
            (v + p, va + p.abs())
        })
}

/// The derived band of a `k`-value row whose absolute products sum to `v_abs`.
fn nvfp4_block_band(k: usize, v_abs: f64) -> f64 {
    let nb = k / 64;
    let n = (nb / 2 + 2 * (nb % 2) + 3) as f64;
    let u = 2f64.powi(-24);
    (n * u / (1.0 - n * u) + k as f64 * 2f64.powi(-53)) * v_abs
}

/// The dot at every length of [`NVFP4_BLOCK_KS`], on random rows and the end rows, against each column: within the derived
/// band of the exact f64 dot, and `dot_row` (the AVX2 kernel on x86_64) the scalar mirror's bits.
#[test]
fn nvfp4_dot_is_the_exact_dot_within_its_band() {
    let mut rng = Lcg(0x4E_F4);
    let (mut cells, mut worst) = (0usize, 0.0f64);
    for k in NVFP4_BLOCK_KS {
        let nb = k / 64;
        let row_bytes = nb * 36;
        let mut rows = nvfp4_block_rows(&mut rng, 6, nb);
        rows.extend(nvfp4_block_end_rows(nb).into_iter().flatten());
        let cols = nvfp4_block_columns(&mut rng, k);
        for (r, row) in rows.chunks_exact(row_bytes).enumerate() {
            for (c, col) in cols.iter().enumerate() {
                let got = dot_row(GgmlType::NVFP4, row, col, k).unwrap();
                let (v, v_abs) = nvfp4_block_exact(row, col, k);
                let (err, band) = ((f64::from(got) - v).abs(), nvfp4_block_band(k, v_abs));
                assert!(
                    err <= band,
                    "k = {k}, row {r}, column {c}: dot {got:e} is {err:e} from the exact {v:e}, band {band:e}"
                );
                if band > 0.0 {
                    worst = worst.max(err / band);
                }
                let mirror = dot_row_scalar(GgmlType::NVFP4, row, col, k).unwrap();
                assert_eq!(
                    got.to_bits(),
                    mirror.to_bits(),
                    "k = {k}, row {r}, column {c}: dot {got:e} against the scalar mirror {mirror:e}"
                );
                cells += 1;
            }
        }
    }
    eprintln!("nvfp4 dot: {cells} cells within the band, worst err/band {worst:.3}");
}

/// Every byte in every sub-block position of a block (and of a lone tail block): a UE4M3 scale gives the exact dot of
/// its decode (`ue4m3_to_f32_half`), a byte that is no scale (bit 7 set, 0x7F) makes the row's value NaN.
#[test]
fn nvfp4_scale_bytes_decode_as_ue4m3_and_a_non_scale_is_nan() {
    for nb in [1usize, 2] {
        let k = 64 * nb;
        let mut col = vec![0u8; col_bytes(GgmlType::NVFP4, k)];
        let x: Vec<f32> = (0..k).map(|i| (i as f32 - 40.0) / 17.0).collect();
        quantize_col(GgmlType::NVFP4, &x, &mut col);
        for sub in 0..4 * nb {
            for byte in 0..=255u8 {
                let mut row = Vec::new();
                for b in 0..nb {
                    let mut d = [0x38u8; 4];
                    if sub / 4 == b {
                        d[sub % 4] = byte;
                    }
                    row.extend(nvfp4_block(d, |i| (i as u8).wrapping_mul(0x35) ^ 0x71));
                }
                let got = dot_row(GgmlType::NVFP4, &row, &col, k).unwrap();
                if gguf::quant::ue4m3_to_f32_half(byte).is_err() {
                    assert!(
                        got.is_nan(),
                        "nb {nb} sub-block {sub} scale {byte:#04x}: {got}"
                    );
                } else {
                    let (v, v_abs) = nvfp4_block_exact(&row, &col, k);
                    let (err, band) = ((f64::from(got) - v).abs(), nvfp4_block_band(k, v_abs));
                    assert!(
                        err <= band,
                        "nb {nb} sub-block {sub} scale {byte:#04x}: dot {got:e} is {err:e} from {v:e}, band {band:e}"
                    );
                }
            }
        }
    }
}

/// The column's bytes, the row's block contract, the refusals by name, and a tile call (NVFP4 has no tile kernel) giving
/// each column `dot_row`'s value.
#[test]
fn nvfp4_shapes_refusals_and_tile_fallback() {
    let ty = GgmlType::NVFP4;
    assert!(supports(ty));
    assert_eq!(activation_format(ty), Ok(Some(ActivationFormat::Q8_2X4)));
    assert_eq!(qdot::k_granularity(ty), 64);
    assert!(qdot::fuses(ty, 64) && !qdot::fuses(ty, 96));
    assert!(!qdot::has_tile(ty));
    // 64-value blocks, 36 bytes each; the column is q8_2_x4 groups of 128 values and q8_2 tails of 32.
    assert_eq!(col_bytes(ty, 2048), 144 * 16);
    assert_eq!(col_bytes(ty, 192), 144 + 2 * 36);
    assert_eq!(col_bytes(ty, 64), 2 * 36);
    let (wrow, acol) = (vec![0u8; 36 * 32], vec![0u8; 144 * 16]);
    assert_eq!(
        dot_row(ty, &wrow, &acol, 96),
        Err(QdotError::UnalignedK { k: 96, gran: 64 })
    );
    assert_eq!(
        dot_row(ty, &wrow[..36 * 32 - 1], &acol, 2048),
        Err(QdotError::ShortWeightRow {
            have: 36 * 32 - 1,
            need: 36 * 32,
            k: 2048
        })
    );
    assert_eq!(
        dot_row(ty, &wrow, &acol[..144 * 16 - 1], 2048),
        Err(QdotError::ShortActivationCol {
            have: 144 * 16 - 1,
            need: 144 * 16,
            k: 2048
        })
    );
    assert_eq!(dot_row(ty, &wrow, &acol, 2048).unwrap(), 0.0);
    let mut rng = Lcg(0x4E_71);
    for k in [64, 192, 2048] {
        let row = nvfp4_block_rows(&mut rng, 1, k / 64);
        let cols = nvfp4_block_columns(&mut rng, k);
        for c in 1..=qdot::TILE_COLS {
            let refs: Vec<&[u8]> = cols[..c].iter().map(Vec::as_slice).collect();
            let mut out = vec![f32::NAN; c];
            qdot::dot_row_cols(ty, &row, &refs, k, &mut out).unwrap();
            for (j, (&got, col)) in out.iter().zip(&refs).enumerate() {
                let want = dot_row(ty, &row, col, k).unwrap();
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "k = {k}, c = {c}, column {j}"
                );
            }
        }
    }
}

/// The AVX2 kernel is the scalar mirror bit for bit on any column bytes a q8_2_x4 column can hold — every code, -128
/// among them, finite bf16 scales, any sums — and not only the encoder's.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_nvfp4_avx2_is_its_scalar_mirror_on_any_column_bytes() {
    let mut rng = Lcg(0x4E_AA);
    let mut cells = 0usize;
    for k in NVFP4_BLOCK_KS {
        let nb = k / 64;
        let rows = nvfp4_block_rows(&mut rng, 8, nb);
        let (groups, tails) = (nb / 2, 2 * (nb % 2));
        for _ in 0..8 {
            let mut col: Vec<u8> = (0..col_bytes(GgmlType::NVFP4, k))
                .map(|_| (rng.next_u32() >> 9) as u8)
                .collect();
            // A bf16 scale with its exponent's top bit clear is finite and at most 2.
            for g in 0..groups {
                for j in 0..4 {
                    col[g * 144 + 2 * j + 1] &= 0xBF;
                }
            }
            for t in 0..tails {
                col[groups * 144 + t * 36 + 1] &= 0xBF;
            }
            for row in rows.chunks_exact(nb * 36) {
                let avx2 = dot_row_avx2(GgmlType::NVFP4, row, &col, k).unwrap();
                let mirror = dot_row_scalar(GgmlType::NVFP4, row, &col, k).unwrap();
                assert!(avx2.is_finite(), "k = {k}: {avx2}");
                assert_eq!(
                    avx2.to_bits(),
                    mirror.to_bits(),
                    "k = {k}: AVX2 {avx2:e} against the mirror {mirror:e}"
                );
                cells += 1;
            }
        }
    }
    eprintln!("nvfp4 avx2: {cells} cells bit-identical to the scalar mirror");
}

// ------------------------------------------------- NVFP4 rows x Q8_2_X4
// bloomery's NVFP4 rows ([`GgmlType::NVFP4_ROWS`]): ModelOpt's groups of 16 values, a UE4M3 scale each,
// value `2i` the low nibble of code byte `i`. One row is read as two planes (`dot_nvfp4_planes_row`) or as
// 9-byte groups (`dot_row`); the reference is `quant::Nvfp4Rows`, the engine's own decode.

/// Row widths: one block, a group, a group and a tail, the model widths and one with a long run of groups and a tail.
const NVFP4_KS: [usize; 10] = [32, 64, 96, 128, 160, 224, 512, 3072, 4128, 6144];

/// f32's unit roundoff.
const NVFP4_U: f64 = 1.0 / 16_777_216.0;

/// A scale byte drawn over the format: zero, the subnormals, the smallest normal, one, the largest scale, or any scale.
fn nvfp4_scale_byte(rng: &mut Lcg) -> u8 {
    match rng.next_u32() % 8 {
        0 => 0x00,
        1 => 1 + (rng.next_u32() % 7) as u8,
        2 => 0x08,
        3 => 0x38,
        4 => 0x7E,
        _ => (rng.next_u32() % 0x7F) as u8,
    }
}

/// A row's planes: every nibble drawn (negative zero and both extreme codes among them), a scale a group.
fn nvfp4_planes(rng: &mut Lcg, k: usize) -> (Vec<u8>, Vec<u8>) {
    let codes = (0..k / 2).map(|_| rng.next_u32() as u8).collect();
    let scales = (0..k / 16).map(|_| nvfp4_scale_byte(rng)).collect();
    (codes, scales)
}

/// The rows at the format's ends: the largest value everywhere, the most negative, negative zero, and a zero scale.
fn nvfp4_end_rows(k: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let planes = |code: u8, scale: u8| (vec![code; k / 2], vec![scale; k / 16]);
    vec![
        planes(0x77, 0x7E),
        planes(0xFF, 0x7E),
        planes(0x88, 0x38),
        planes(0x35, 0x00),
    ]
}

/// The planes as the row form: per group the scale byte, then its 8 code bytes.
fn nvfp4_interleave(codes: &[u8], scales: &[u8]) -> Vec<u8> {
    scales
        .iter()
        .zip(codes.chunks(8))
        .flat_map(|(&s, c)| std::iter::once(s).chain(c.iter().copied()))
        .collect()
}

/// The planes as mainline's blocks (`k` a multiple of 64): per block its four scales, then the 32 code bytes, each
/// group's eight bytes in split order (value `j` and `j + 8` the low and high nibble of byte `j`) where the planes
/// hold pair order (value `2i` and `2i + 1`).
fn nvfp4_to_blocks(codes: &[u8], scales: &[u8]) -> Vec<u8> {
    let value = |g: usize, v: usize| {
        let byte = codes[8 * g + v / 2];
        if v.is_multiple_of(2) {
            byte & 0xF
        } else {
            byte >> 4
        }
    };
    scales
        .chunks(4)
        .enumerate()
        .flat_map(|(b, d)| {
            let qs = (0..32).map(move |i| {
                let (g, j) = (4 * b + i / 8, i % 8);
                value(g, j) | (value(g, j + 8) << 4)
            });
            d.iter().copied().chain(qs).collect::<Vec<u8>>()
        })
        .collect()
}

/// Quantized columns of one width: values of several amplitudes, a spike over small values, and zeros.
fn nvfp4_columns(rng: &mut Lcg, k: usize) -> Vec<Vec<u8>> {
    let mut xs: Vec<Vec<f32>> = [1.0f32, 1e-3, 37.0, 3e4]
        .iter()
        .map(|&amp| (0..k).map(|_| rng.unit() * amp).collect())
        .collect();
    let mut spike: Vec<f32> = (0..k).map(|_| rng.unit() * 1e-3).collect();
    spike[k / 3] = 1000.0;
    xs.push(spike);
    xs.push(vec![0.0; k]);
    xs.iter()
        .map(|x| {
            let mut col = vec![0u8; col_bytes(GgmlType::NVFP4_ROWS, k)];
            quantize_col(GgmlType::NVFP4_ROWS, x, &mut col);
            col
        })
        .collect()
}

/// A column that need not come from the encoder: any code bytes (-128 among them, which `sign` wraps), any sums,
/// and a finite bf16 scale a group (at most 4).
#[cfg(target_arch = "x86_64")]
fn nvfp4_raw_column(rng: &mut Lcg, k: usize) -> Vec<u8> {
    let nbg = k / 128;
    let tails = k / 32 - 4 * nbg;
    let mut col: Vec<u8> = (0..col_bytes(GgmlType::NVFP4_ROWS, k))
        .map(|_| rng.next_u32() as u8)
        .collect();
    let mut set_d = |at: usize, rng: &mut Lcg| {
        let bits = ((rng.unit() * 4.0).to_bits() >> 16) as u16;
        col[at..at + 2].copy_from_slice(&bits.to_le_bytes());
    };
    for g in 0..nbg {
        for l in 0..4 {
            set_d(g * 144 + 2 * l, rng);
        }
    }
    for t in 0..tails {
        set_d(nbg * 144 + 36 * t, rng);
    }
    col
}

/// The exact f64 dot of the row's decode (`Nvfp4Rows`, tensor scale 1) with the column's restored
/// values, and the band the f32 kernel stays in, derived before the run. A leaf is what one fmadd
/// adds to its lane: an integer sum (at most 16·12·128, exact in f32) times a scale that is a 4-bit
/// significand times a bf16 activation scale (12 bits, exact), so each leaf is exact. In an x4
/// group the lanes are combined first, so a leaf is one 16-value group's sum; in a tail block a
/// lane holds four values, so a leaf is four. A lane takes one fmadd a group and one a tail block
/// (`nb / 4 + nb % 4`), then three reduction levels: `n = nb / 4 + nb % 4 + 3` roundings, the band
/// `γ_n · Σ|leaf|`. The f64 reference's own sum adds `k · 2^-53 · Σ|w · a|`.
fn nvfp4_exact(codes: &[u8], scales: &[u8], acol: &[u8], k: usize) -> (f64, f64) {
    let mut w = vec![0.0f32; k];
    Nvfp4Rows::new(1.0)
        .decode_planes(codes, scales, &mut w)
        .expect("a row of valid scales decodes");
    let a = restore_q82x4(acol);
    let (mut dot, mut leaves) = (0.0f64, 0.0f64);
    let grouped = k / 128 * 128;
    let terms: Vec<f64> = w.iter().zip(&a).map(|(&x, &y)| f64::from(x) * y).collect();
    for leaf in terms[..grouped]
        .chunks(16)
        .chain(terms[grouped..].chunks(4))
    {
        let sum: f64 = leaf.iter().sum();
        dot += sum;
        leaves += sum.abs();
    }
    let wa: f64 = terms.iter().map(|t| t.abs()).sum();
    let nb = k / 32;
    let n = (nb / 4 + nb % 4 + 3) as f64;
    let band = n * NVFP4_U / (1.0 - n * NVFP4_U) * leaves + k as f64 * 2f64.powi(-53) * wa;
    (dot, band)
}

/// Every entry that reads one NVFP4 row, over every row, column and width, against [`nvfp4_exact`]:
/// the plane kernel and its scalar mirror, and `dot_row` on the row form and (at multiples of 64) on
/// mainline's block form, which must give the plane kernel's bits (three layouts, one float order), and the
/// plane tile over the row's columns, whose every value is the plane kernel's. A wrong nibble order, scale lane,
/// tail or scale decode moves whole groups, far past the band.
#[test]
fn nvfp4_rows_dot_is_the_exact_dot_within_its_band() {
    let mut rng = Lcg(0x4e56_4650_3420);
    let mut worst = 0.0f64;
    for &k in &NVFP4_KS {
        let mut rows: Vec<(Vec<u8>, Vec<u8>)> =
            (0..10).map(|_| nvfp4_planes(&mut rng, k)).collect();
        rows.extend(nvfp4_end_rows(k));
        let cols = nvfp4_columns(&mut rng, k);
        for (r, (codes, scales)) in rows.iter().enumerate() {
            let row_form = nvfp4_interleave(codes, scales);
            let refs: Vec<&[u8]> = cols.iter().map(Vec::as_slice).collect();
            let mut tile = vec![f32::NAN; refs.len()];
            qdot::dot_nvfp4_planes_row_cols(codes, scales, &refs, k, &mut tile);
            for (c, acol) in cols.iter().enumerate() {
                let at = format!("k = {k} row {r} column {c}");
                let (exact, band) = nvfp4_exact(codes, scales, acol, k);
                let planes = qdot::dot_nvfp4_planes_row(codes, scales, acol, k);
                let scalar = qdot::dot_nvfp4_planes_row_scalar(codes, scales, acol, k);
                let via_row = dot_row(GgmlType::NVFP4_ROWS, &row_form, acol, k).unwrap();
                let via_row_scalar =
                    dot_row_scalar(GgmlType::NVFP4_ROWS, &row_form, acol, k).unwrap();
                for (path, got) in [("planes", planes), ("scalar", scalar), ("dot_row", via_row)] {
                    let err = (f64::from(got) - exact).abs();
                    assert!(
                        err <= band,
                        "{at}: {path} {got:e} is {err:e} from the exact {exact:e}, band {band:e}"
                    );
                    if band > 0.0 {
                        worst = worst.max(err / band);
                    }
                }
                if k % 64 == 0 {
                    let blocks = nvfp4_to_blocks(codes, scales);
                    let via_blocks = dot_row(GgmlType::NVFP4, &blocks, acol, k).unwrap();
                    let via_blocks_scalar =
                        dot_row_scalar(GgmlType::NVFP4, &blocks, acol, k).unwrap();
                    assert_eq!(
                        planes.to_bits(),
                        via_blocks.to_bits(),
                        "{at}: mainline's block form against the planes"
                    );
                    assert_eq!(
                        scalar.to_bits(),
                        via_blocks_scalar.to_bits(),
                        "{at}: the mirror's block form"
                    );
                }
                assert_eq!(
                    planes.to_bits(),
                    scalar.to_bits(),
                    "{at}: kernel against its mirror"
                );
                assert_eq!(
                    planes.to_bits(),
                    tile[c].to_bits(),
                    "{at}: the plane tile against the planes"
                );
                assert_eq!(
                    planes.to_bits(),
                    via_row.to_bits(),
                    "{at}: the row form against the planes"
                );
                assert_eq!(
                    scalar.to_bits(),
                    via_row_scalar.to_bits(),
                    "{at}: the mirror's two forms"
                );
            }
        }
    }
    println!("nvfp4 rows: worst err/band {worst:.3}");
    assert!(worst < 1.0);
}

/// Pair order against mainline's split order, which the three layouts split between them: code bytes
/// all `0x42` (low nibble the doubled value 2, high nibble 4), scales 1 and 2 alternating by group, a
/// column that is 1 on even values and 0 on odd. At scale `s` a 16-value group in pair order puts
/// its low nibbles on the even values, eight of `1 · s`: `8 s a`. In split order the low nibbles
/// are values 0 to 7 and the high ones 8 to 15, so the even values are four of `1 · s` and four of
/// `2 · s`: `12 s a`. Over 64 values (scales 1, 2, 1, 2) the planes and the row form give
/// `(8 + 16) · 2 = 48 a` and mainline's block, on the same bytes, `(12 + 24) · 2 = 72 a`.
#[test]
fn nvfp4_layouts_read_their_own_nibble_order() {
    let k = 64;
    let d = [0x38u8, 0x40, 0x38, 0x40];
    let (codes, scales) = (vec![0x42u8; 32], d.to_vec());
    let x: Vec<f32> = (0..k).map(|v| if v % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let mut col = vec![0u8; col_bytes(GgmlType::NVFP4_ROWS, k)];
    quantize_col(GgmlType::NVFP4_ROWS, &x, &mut col);
    let a = restore_q82x4(&col)[0];
    assert!(a > 0.0);

    let (exact, band) = nvfp4_exact(&codes, &scales, &col, k);
    assert!(
        (exact / a - 48.0).abs() < 1e-9,
        "the reference decode reads pair order: {} a",
        exact / a
    );
    let row = nvfp4_interleave(&codes, &scales);
    for (path, got) in [
        (
            "planes",
            qdot::dot_nvfp4_planes_row(&codes, &scales, &col, k),
        ),
        (
            "scalar",
            qdot::dot_nvfp4_planes_row_scalar(&codes, &scales, &col, k),
        ),
        (
            "rows",
            dot_row(GgmlType::NVFP4_ROWS, &row, &col, k).unwrap(),
        ),
    ] {
        assert!(
            (f64::from(got) - 48.0 * a).abs() <= band,
            "{path}: {got:e}, pair order wants {:e}, split order {:e}",
            48.0 * a,
            72.0 * a
        );
    }

    let block = nvfp4_block(d, |_| 0x42);
    let (exact, v_abs) = nvfp4_block_exact(&block, &col, k);
    assert!(
        (exact / a - 72.0).abs() < 1e-9,
        "mainline's decode reads split order: {} a",
        exact / a
    );
    let got = dot_row(GgmlType::NVFP4, &block, &col, k).unwrap();
    assert!(
        (f64::from(got) - 72.0 * a).abs() <= nvfp4_block_band(k, v_abs),
        "block: {got:e}, split order wants {:e}, pair order {:e}",
        72.0 * a,
        48.0 * a
    );
}

/// Every byte at every scale position, in a row of one group (eight scales) and one with a tail
/// (ten): a byte up to 0x7E is a scale and the dot stays within its band, any other byte (bit 7
/// set, or 0x7F, E4M3's NaN) makes the dot NaN exactly where `Nvfp4Rows` refuses the row, through
/// the plane kernel, its mirror, its tile and every form.
#[test]
fn nvfp4_rows_scale_bytes_poison_the_row_when_they_are_no_scale() {
    let mut rng = Lcg(0x0053_4341_4c45);
    for k in [128usize, 160] {
        let codes: Vec<u8> = (0..k / 2).map(|_| rng.next_u32() as u8).collect();
        let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
        let mut col = vec![0u8; col_bytes(GgmlType::NVFP4_ROWS, k)];
        quantize_col(GgmlType::NVFP4_ROWS, &x, &mut col);
        for at in 0..k / 16 {
            for byte in 0..=255u8 {
                let mut scales = vec![0x38u8; k / 16];
                scales[at] = byte;
                let row_form = nvfp4_interleave(&codes, &scales);
                let mut got = vec![
                    qdot::dot_nvfp4_planes_row(&codes, &scales, &col, k),
                    qdot::dot_nvfp4_planes_row_scalar(&codes, &scales, &col, k),
                    dot_row(GgmlType::NVFP4_ROWS, &row_form, &col, k).unwrap(),
                    dot_row_scalar(GgmlType::NVFP4_ROWS, &row_form, &col, k).unwrap(),
                ];
                if k % 64 == 0 {
                    let blocks = nvfp4_to_blocks(&codes, &scales);
                    got.push(dot_row(GgmlType::NVFP4, &blocks, &col, k).unwrap());
                    got.push(dot_row_scalar(GgmlType::NVFP4, &blocks, &col, k).unwrap());
                }
                let mut tile = [0.0f32; 2];
                qdot::dot_nvfp4_planes_row_cols(
                    &codes,
                    &scales,
                    &[&col[..], &col[..]],
                    k,
                    &mut tile,
                );
                got.extend(tile);
                let mut w = vec![0.0f32; k];
                let refused = Nvfp4Rows::new(1.0)
                    .decode_planes(&codes, &scales, &mut w)
                    .is_err();
                assert_eq!(
                    refused,
                    byte > 0x7E,
                    "k = {k} scale {at} byte {byte:#04x}: the decode's refusal"
                );
                if refused {
                    assert!(
                        got.iter().all(|g| g.is_nan()),
                        "k = {k} scale {at} byte {byte:#04x}: {got:?} is no NaN"
                    );
                } else {
                    let (exact, band) = nvfp4_exact(&codes, &scales, &col, k);
                    for g in got {
                        assert!(
                            (f64::from(g) - exact).abs() <= band,
                            "k = {k} scale {at} byte {byte:#04x}: {g:e} against {exact:e}, band {band:e}"
                        );
                    }
                }
            }
        }
    }
}

/// The type's entries and its row agree: Q8_2_X4 columns of MXFP4's shape and bytes, 32-value granularity, no
/// `dot_row_cols` tile (the planes have their own, `dot_nvfp4_planes_row_cols`), `dot_row`'s named refusals, and
/// the plane entries, which panic by name on a layout or a tile shape they cannot read.
#[test]
fn nvfp4_rows_entries_and_refusals() {
    let ty = GgmlType::NVFP4_ROWS;
    assert!(supports(ty) && qdot::fuses(ty, 32) && !qdot::fuses(ty, 48));
    assert_eq!(qdot::k_granularity(ty), 32);
    assert!(!qdot::has_tile(ty));
    assert_eq!(activation_format(ty), Ok(Some(ActivationFormat::Q8_2X4)));
    let mut rng = Lcg(0x454e_5452);
    for k in [32, 64, 96, 128, 160, 4128] {
        assert_eq!(col_bytes(ty, k), col_bytes(GgmlType::MXFP4, k), "k = {k}");
        let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
        let (mut a, mut b) = (vec![0u8; col_bytes(ty, k)], vec![0u8; col_bytes(ty, k)]);
        quantize_col(ty, &x, &mut a);
        quantize_col(GgmlType::MXFP4, &x, &mut b);
        assert_eq!(a, b, "k = {k}: MXFP4's column");
        if k % 64 == 0 {
            let mut c = vec![0u8; col_bytes(GgmlType::NVFP4, k)];
            quantize_col(GgmlType::NVFP4, &x, &mut c);
            assert_eq!(a, c, "k = {k}: mainline's block takes the same column");
        }
    }
    let (codes, scales) = nvfp4_planes(&mut rng, 64);
    let row = nvfp4_interleave(&codes, &scales);
    let col = nvfp4_columns(&mut rng, 64).remove(0);
    assert_eq!(
        dot_row(ty, &row, &col, 48),
        Err(QdotError::UnalignedK { k: 48, gran: 32 })
    );
    assert_eq!(
        dot_row(ty, &row[..35], &col, 64),
        Err(QdotError::ShortWeightRow {
            have: 35,
            need: 36,
            k: 64
        })
    );
    assert_eq!(
        dot_row(ty, &row, &col[..col.len() - 1], 64),
        Err(QdotError::ShortActivationCol {
            have: col.len() - 1,
            need: col.len(),
            k: 64
        })
    );
    let outs = [&col[..], &col[..]];
    let mut cols = [0.0f32; 2];
    qdot::dot_row_cols(ty, &row, &outs, 64, &mut cols).unwrap();
    assert_eq!(cols, [dot_row(ty, &row, &col, 64).unwrap(); 2]);

    let panic_text = |f: &dyn Fn() -> f32| -> String {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        match r.expect_err("the plane entry must refuse") {
            e if e.is::<String>() => *e.downcast::<String>().unwrap(),
            e => (*e.downcast::<&str>().unwrap()).to_owned(),
        }
    };
    let run = |c: &[u8], s: &[u8], a: &[u8], k: usize| qdot::dot_nvfp4_planes_row(c, s, a, k);
    for (what, text) in [
        (
            "an unaligned k",
            panic_text(&|| run(&codes, &scales, &col, 48)),
        ),
        (
            "a short code plane",
            panic_text(&|| run(&codes[..31], &scales, &col, 64)),
        ),
        (
            "a short scale plane",
            panic_text(&|| run(&codes, &scales[..3], &col, 64)),
        ),
        (
            "a short column",
            panic_text(&|| run(&codes, &scales, &col[..col.len() - 1], 64)),
        ),
    ] {
        assert!(text.contains("dot_nvfp4_planes_row"), "{what}: {text}");
    }

    // The tile entry: a column count outside 1..=TILE_COLS and an output of another length are its own refusals, a
    // plane or any column's shape the plane entry's.
    let tile = |cols: &[&[u8]], outs: usize, k: usize, codes: &[u8], scales: &[u8]| {
        let mut out = vec![0.0f32; outs];
        qdot::dot_nvfp4_planes_row_cols(codes, scales, cols, k, &mut out);
        out[0]
    };
    let nine = vec![&col[..]; qdot::TILE_COLS + 1];
    let short = &col[..col.len() - 1];
    for (what, text, name) in [
        (
            "no columns",
            panic_text(&|| tile(&[], 0, 64, &codes, &scales)),
            "dot_nvfp4_planes_row_cols",
        ),
        (
            "more than TILE_COLS columns",
            panic_text(&|| tile(&nine, nine.len(), 64, &codes, &scales)),
            "dot_nvfp4_planes_row_cols",
        ),
        (
            "an output of another length",
            panic_text(&|| tile(&[&col[..], &col[..]], 3, 64, &codes, &scales)),
            "dot_nvfp4_planes_row_cols",
        ),
        (
            "a short second column",
            panic_text(&|| tile(&[&col[..], short], 2, 64, &codes, &scales)),
            "dot_nvfp4_planes_row",
        ),
        (
            "a short code plane",
            panic_text(&|| tile(&[&col[..], &col[..]], 2, 64, &codes[..31], &scales)),
            "dot_nvfp4_planes_row",
        ),
        (
            "an unaligned k",
            panic_text(&|| tile(&[&col[..], &col[..]], 2, 48, &codes, &scales)),
            "dot_nvfp4_planes_row",
        ),
    ] {
        assert!(text.contains(name), "{what}: {text}");
    }

    // The rows entry: rows past either plane are its own refusal, a column's shape the plane entry's.
    let rows_entry = |rows: std::ops::Range<usize>, cols: &[&[u8]], k: usize| {
        let mut last = 0.0f32;
        qdot::dot_nvfp4_planes_rows_cols(
            (&codes, &scales),
            rows,
            k,
            (cols.len(), |j| cols[j]),
            |_, _, v| last = v,
        );
        last
    };
    for (what, text, name) in [
        (
            "a row past the planes",
            panic_text(&|| rows_entry(0..2, &[&col[..], &col[..]], 64)),
            "dot_nvfp4_planes_rows_cols",
        ),
        (
            "a short second column",
            panic_text(&|| rows_entry(0..1, &[&col[..], short], 64)),
            "dot_nvfp4_planes_row",
        ),
    ] {
        assert!(text.contains(name), "{what}: {text}");
    }
}

/// The AVX2 plane kernel against its mirror, bit for bit, on columns that need not come from the
/// encoder (any codes, including -128, which `sign` wraps; finite bf16 scales; any sums) and on
/// encoder columns at the model widths, with every scale of the format, and the row form's kernel
/// and mainline's block form's against the planes'. A row holding a byte that is no scale is NaN in all.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_nvfp4_rows_avx2_is_its_scalar_mirror() {
    let mut rng = Lcg(0x4156_5832);
    let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    for &k in &NVFP4_KS {
        let mut cols = nvfp4_columns(&mut rng, k);
        cols.extend((0..4).map(|_| nvfp4_raw_column(&mut rng, k)));
        let mut rows: Vec<(Vec<u8>, Vec<u8>)> =
            (0..12).map(|_| nvfp4_planes(&mut rng, k)).collect();
        rows.extend(nvfp4_end_rows(k));
        let mut bad = nvfp4_planes(&mut rng, k);
        let last = bad.1.len() - 1;
        bad.1[last] = 0x7F;
        rows.push(bad);
        for (r, (codes, scales)) in rows.iter().enumerate() {
            let row_form = nvfp4_interleave(codes, scales);
            for (c, acol) in cols.iter().enumerate() {
                let at = format!("k = {k} row {r} column {c}");
                let avx = qdot::dot_nvfp4_planes_row_avx2(codes, scales, acol, k);
                let mirror = qdot::dot_nvfp4_planes_row_scalar(codes, scales, acol, k);
                let form = dot_row_avx2(GgmlType::NVFP4_ROWS, &row_form, acol, k).unwrap();
                assert!(
                    same(avx, mirror),
                    "{at}: AVX2 {avx:e} against the mirror {mirror:e}"
                );
                assert!(
                    same(form, avx),
                    "{at}: the row form {form:e} against the planes {avx:e}"
                );
                if k % 64 == 0 {
                    let blocks = nvfp4_to_blocks(codes, scales);
                    let block = dot_row_avx2(GgmlType::NVFP4, &blocks, acol, k).unwrap();
                    assert!(
                        same(block, avx),
                        "{at}: mainline's block {block:e} against the planes {avx:e}"
                    );
                }
            }
        }
    }
}

/// The plane tile against the plane kernel, bit for bit: at every width of [`NVFP4_KS`], random rows, the format's
/// end rows and rows holding a byte that is no scale (NaN in every column; one in the first group, one in the last
/// scale), against the encoder's columns and raw ones (any codes but -128, which the tile refuses by name in
/// `hw_nvfp4_planes_tile_refuses_code_minus_128`; any sums), at every column count `1..=TILE_COLS` with the columns
/// rotated so each tile slot meets each column. A slot holds a payload no dot has until the tile
/// writes it, so a slot left unwritten, one that reads another's column, a tail, a scale lane or a poison check done
/// wrong shows as a differing bit; the band to the exact dot is `nvfp4_rows_dot_is_the_exact_dot_within_its_band`'s.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_nvfp4_planes_tile_matches_the_column_kernel() {
    assert!(
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        "the tile runs with AVX2 and FMA only; without them this clause would compare the column kernel with itself"
    );
    const UNWRITTEN: f32 = f32::from_bits(0x7FC0_0001);
    let mut rng = Lcg(0x4e56_5449_4c45);
    let mut calls = 0usize;
    for &k in &NVFP4_KS {
        let mut cols = nvfp4_columns(&mut rng, k);
        cols.extend((0..4).map(|_| nvfp4_tile_raw_column(&mut rng, k)));
        let mut rows: Vec<(Vec<u8>, Vec<u8>)> = (0..6).map(|_| nvfp4_planes(&mut rng, k)).collect();
        rows.extend(nvfp4_end_rows(k));
        for (at, byte) in [(0, 0x80), (k / 16 - 1, 0x7F)] {
            let mut bad = nvfp4_planes(&mut rng, k);
            bad.1[at] = byte;
            rows.push(bad);
        }
        for (r, (codes, scales)) in rows.iter().enumerate() {
            let want: Vec<f32> = cols
                .iter()
                .map(|a| qdot::dot_nvfp4_planes_row_avx2(codes, scales, a, k))
                .collect();
            for c in 1..=qdot::TILE_COLS {
                for shift in 0..cols.len() {
                    let pick: Vec<usize> = (0..c).map(|j| (shift + j) % cols.len()).collect();
                    let refs: Vec<&[u8]> = pick.iter().map(|&p| cols[p].as_slice()).collect();
                    let mut out = vec![UNWRITTEN; c];
                    qdot::dot_nvfp4_planes_row_cols(codes, scales, &refs, k, &mut out);
                    for (j, (&got, &p)) in out.iter().zip(&pick).enumerate() {
                        assert_eq!(
                            got.to_bits(),
                            want[p].to_bits(),
                            "k = {k} row {r} c = {c} shift {shift} slot {j} (column {p}): tile {got:e} \
                             against the column kernel {:e}",
                            want[p]
                        );
                    }
                    calls += 1;
                }
            }
        }
    }
    eprintln!("nvfp4 planes tile: {calls} calls, c = 1..=8, bit-identical to the column kernel");
}

/// The byte offsets of a Q8_2_X4 column's codes at `k`: each x4 group's 128 after its 16-byte head, each tail
/// block's 32 after its 4-byte head.
#[cfg(target_arch = "x86_64")]
fn nvfp4_code_bytes(k: usize) -> impl Iterator<Item = usize> {
    let nbg = k / 128;
    let tails = k / 32 - 4 * nbg;
    let groups = (0..nbg).flat_map(|g| g * 144 + 16..g * 144 + 144);
    let tail = (0..tails).flat_map(move |t| nbg * 144 + 36 * t + 4..nbg * 144 + 36 * t + 36);
    groups.chain(tail)
}

/// [`nvfp4_raw_column`] with each code -128 made -127: a column the tile takes, any other code and any sums.
#[cfg(target_arch = "x86_64")]
fn nvfp4_tile_raw_column(rng: &mut Lcg, k: usize) -> Vec<u8> {
    let mut col = nvfp4_raw_column(rng, k);
    for at in nvfp4_code_bytes(k) {
        if col[at] == 0x80 {
            col[at] = 0x81;
        }
    }
    col
}

/// The rows entry against the plane kernel, bit for bit, as the host union calls it: a run of rows past one row
/// block (random rows, the end rows, a row holding a byte that is no scale), from a first row past 0, against column
/// counts of one run (2, 8), of two balanced runs (9, 16) and of three (17), each column an encoder's or a raw one's
/// through the column accessor, every value to its own (row, column) cell. A run's columns read with another run's
/// preps, a prep built of one row block and read by the next, a run cut that loses or repeats a column, or a value
/// written to another cell shows as a differing bit or a cell left unwritten.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_nvfp4_planes_rows_match_the_column_kernel() {
    assert!(
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        "the tile runs with AVX2 and FMA only; without them this clause would compare the column kernel with itself"
    );
    const UNWRITTEN: f32 = f32::from_bits(0x7FC0_0001);
    const ROWS: usize = 150;
    const FIRST: usize = 3;
    let mut rng = Lcg(0x524f_5753_4e56);
    for k in [128usize, 3072, 4128] {
        let mut cols = nvfp4_columns(&mut rng, k);
        cols.extend((0..4).map(|_| nvfp4_tile_raw_column(&mut rng, k)));
        let mut rows: Vec<(Vec<u8>, Vec<u8>)> =
            (0..ROWS - 5).map(|_| nvfp4_planes(&mut rng, k)).collect();
        rows.extend(nvfp4_end_rows(k));
        let mut bad = nvfp4_planes(&mut rng, k);
        bad.1[k / 32] = 0x7F;
        rows.insert(ROWS / 2, bad);
        let codes: Vec<u8> = rows.iter().flat_map(|r| r.0.iter().copied()).collect();
        let scales: Vec<u8> = rows.iter().flat_map(|r| r.1.iter().copied()).collect();
        for ncols in [2usize, 8, 9, 16, 17] {
            let pick: Vec<usize> = (0..ncols).map(|j| (3 * j + ncols) % cols.len()).collect();
            let mut out = vec![UNWRITTEN; ROWS * ncols];
            qdot::dot_nvfp4_planes_rows_cols(
                (&codes, &scales),
                FIRST..ROWS,
                k,
                (ncols, |j| cols[pick[j]].as_slice()),
                |r, j, v| {
                    assert!(
                        out[j * ROWS + r].to_bits() == UNWRITTEN.to_bits(),
                        "k = {k} ncols {ncols}: cell (row {r}, column {j}) written twice"
                    );
                    out[j * ROWS + r] = v;
                },
            );
            for (r, (c, sc)) in rows.iter().enumerate() {
                for (j, &p) in pick.iter().enumerate() {
                    let got = out[j * ROWS + r];
                    if r < FIRST {
                        assert_eq!(
                            got.to_bits(),
                            UNWRITTEN.to_bits(),
                            "k = {k} ncols {ncols}: row {r} lies before the call's rows and was written"
                        );
                        continue;
                    }
                    let want = qdot::dot_nvfp4_planes_row_avx2(c, sc, &cols[p], k);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "k = {k} ncols {ncols} row {r} slot {j} (column {p}): rows entry {got:e} against the column \
                         kernel {want:e}"
                    );
                }
            }
        }
    }
}

/// The tile refuses a column holding the code -128 by name, in an x4 group or in a tail block: no encoder writes it,
/// the one-column kernel's sign fold wraps it to -128 where its product needs +128, and the tile's offset dot would
/// read it as itself, so the two would part on that column without a word.
#[cfg(target_arch = "x86_64")]
#[test]
#[ignore = "hw: needs the box (AVX2)"]
fn hw_nvfp4_planes_tile_refuses_code_minus_128() {
    assert!(
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        "the tile runs with AVX2 and FMA only"
    );
    let mut rng = Lcg(0x4d49_4e31_3238);
    let k = 160;
    let (codes, scales) = nvfp4_planes(&mut rng, k);
    let clean = nvfp4_columns(&mut rng, k).remove(0);
    for (where_, at) in [("an x4 group", 16 + 77), ("a tail block", 144 + 4 + 9)] {
        let mut col = clean.clone();
        col[at] = 0x80;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out = [0.0f32; 2];
            qdot::dot_nvfp4_planes_row_cols(&codes, &scales, &[&clean[..], &col[..]], k, &mut out);
        }));
        let text = match r.expect_err("the tile must refuse a column holding -128") {
            e if e.is::<String>() => *e.downcast::<String>().unwrap(),
            e => (*e.downcast::<&str>().unwrap()).to_owned(),
        };
        assert!(
            text.contains("dot_nvfp4_planes_rows_cols")
                && text.contains("column 1 holds the code -128"),
            "-128 in {where_}: {text}"
        );
    }
}

// ------------------------------------------------- NVFP4 rows' codes in block order
// `nvfp4_codes_to_blocks` (AVX2 where the machine has it) and its scalar mirror against the
// per-value rule written out here: decode each group's 16 values in the rows' order, then write
// them in the block's order. Lengths of 0, 8, 24 (a tail alone), 32, 40 (a step and a tail) and one
// expert's gate row bytes at K = 6,144.

/// Group `g`'s 16 code values of NVFP4 rows' bytes: value `2i` the low nibble of byte `i`, `2i + 1`
/// its high nibble.
fn nvfp4_rows_values(group: &[u8]) -> [u8; 16] {
    std::array::from_fn(|v| (group[v / 2] >> (4 * (v % 2))) & 0x0f)
}

/// Block-ordered bytes `nvfp4_codes_to_blocks` writes, and its scalar mirror, equal the per-value
/// rule over random bytes; a refusal writes nothing. Mutant: the AVX2 body's `unpacklo_epi64`
/// swapped for `unpackhi_epi64` — every 32-byte step differs, the scalar arm stays green.
#[test]
fn nvfp4_codes_to_blocks_follows_the_value_rule() {
    let mut rng = Rng(0x0DD5_EED5_0000_0004);
    for len in [0usize, 8, 24, 32, 40, 3072] {
        let src = random_bytes(&mut rng, len);
        let mut want = vec![0u8; len];
        for (g, o) in src
            .as_chunks::<8>()
            .0
            .iter()
            .zip(want.as_chunks_mut::<8>().0.iter_mut())
        {
            let v = nvfp4_rows_values(g);
            for (j, b) in o.iter_mut().enumerate() {
                *b = v[j] | (v[j + 8] << 4);
            }
        }
        let mut got = vec![0xA5u8; len];
        qdot::nvfp4_codes_to_blocks(&src, &mut got).unwrap();
        assert!(
            got == want,
            "{len} bytes: the dispatch differs from the value rule at byte {:?}",
            first_diff(&got, &want)
        );
        let mut scalar = vec![0x5Au8; len];
        qdot::nvfp4_codes_to_blocks_scalar(&src, &mut scalar).unwrap();
        assert!(
            scalar == want,
            "{len} bytes: the scalar mirror differs from the value rule at byte {:?}",
            first_diff(&scalar, &want)
        );
    }
    let src = random_bytes(&mut rng, 16);
    let mut dst = [0xA5u8; 16];
    assert_eq!(
        qdot::nvfp4_codes_to_blocks(&src[..12], &mut dst[..12]),
        Err(QdotError::UnalignedK { k: 24, gran: 16 })
    );
    assert_eq!(
        qdot::nvfp4_codes_to_blocks(&src, &mut dst[..8]),
        Err(QdotError::RowGroupBytes {
            buf: "nvfp4 codes destination",
            have: 8,
            need: 16,
        })
    );
    assert!(dst.iter().all(|&b| b == 0xA5), "a refusal wrote into dst");
}
