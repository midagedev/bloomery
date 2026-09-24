//! qdot gates: fused kernels against matmul_q, exact f64 reference, and scalar mirrors.
//!
//! `hw_` prefix: needs the box (the model file and `$BLOOMERY_DATA/ref`),
//! excluded by default, run by `just gate-qdot`. The one pure gate
//! (`rejects_unaligned_k`) runs in the default set.
//!
//! The gates assert with `assert!` (not `debug_assert!`) because they run in release.

use gguf::GgmlType;
use gguf::quant::{dequant_row, quantize_row_q8_k_roundtrip};
use qdot::{
    QdotError, col_bytes, dot_row, dot_row_avx2, dot_row_scalar, quantize_col, quantize_col_scalar,
    supports,
};

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
    // Supported type table check.
    assert!(!supports(GgmlType::F16));
    assert!(supports(GgmlType::Q5_K));
    assert!(supports(GgmlType::Q5_1));
    assert!(supports(GgmlType::IQ3_XXS));
    assert!(supports(GgmlType::MXFP4));
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

/// A q8_2_x4 column as the kernels read it: per 32-value block, the bf16 scale at `2*ir` of its
/// 144-byte group times each i8 code at `16 + 32*ir`. Whole groups only — a K-quant column
/// has no tail blocks.
fn restore_q82x4(acol: &[u8]) -> Vec<f64> {
    assert!(acol.len().is_multiple_of(144), "whole 144-byte groups only");
    let mut out = Vec::with_capacity(acol.len() / 144 * 128);
    for g in acol.as_chunks::<144>().0 {
        for ir in 0..4 {
            let d = f32::from_bits(u32::from(u16::from_le_bytes([g[2 * ir], g[2 * ir + 1]])) << 16);
            out.extend(
                g[16 + 32 * ir..16 + 32 * ir + 32]
                    .iter()
                    .map(|&q| f64::from(d) * f64::from(q as i8)),
            );
        }
    }
    out
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
// The V4-Flash routed-expert formats. The reference model carries neither; the V4-Flash
// file's first data shard holds blk.0.ffn_gate_exps.weight (IQ3_XXS, k = 4096) and
// blk.0.ffn_down_exps.weight (MXFP4, k = 2048). Its tensors are found through the
// header-only inventory and their rows read at `data_base + offset`, as the Q5_K case does.
// Gate A: the AVX2 kernel equals its mirror bit for bit on the tensor's first rows. Gate B:
// on ik's own column the kernel is within 1 ULP of ik's kernel on all 64 dumped rows
// (tools/ref/iq3xxs_ref.cpp, tools/ref/mxfp4_x4_ref.cpp). MXFP4 also carries gate 0, the
// encoder against ik's quantize_row_q8_2_x4; IQ3_XXS cannot: ik's AVX2 q8_K encoder codes
// from the unsigned max (d > 0 always) where ours mirrors ggml's signed-extreme reference,
// so the bytes differ while the products agree.

/// The V4-Flash shard the two gates and their harnesses read; `BLOOMERY_V4_MODEL`
/// overrides it (tools/ref/build-qdot-ref.sh reads the same variable, same default).
fn v4_model() -> String {
    std::env::var("BLOOMERY_V4_MODEL").unwrap_or_else(|_| {
        "/models/DeepSeek-V4-Flash-0731-UD-Q3_K_M/DeepSeek-V4-Flash-0731-UD-Q3_K_M-00002-of-00004.gguf"
            .into()
    })
}

/// The first `ty` tensor of the V4-Flash shard with an aligned k — the scan its
/// harness does — with its first rows and the column `quantize_col` codes from
/// the first k values of the attn_norm-0 dump (the column the harness codes).
struct V4Case {
    ty: GgmlType,
    name: String,
    k: usize,
    row_bytes: usize,
    bytes: Vec<u8>,
    acol: Vec<u8>,
}

impl V4Case {
    fn load(ty: GgmlType, rows: usize) -> V4Case {
        use std::os::unix::fs::FileExt;
        let (gran, block): (usize, usize) = match ty {
            GgmlType::IQ3_XXS => (256, 98),
            GgmlType::MXFP4 => (32, 17),
            _ => panic!("no V4 case for {ty:?}"),
        };
        let path = v4_model();
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
        V4Case {
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

/// IQ3_XXS gate A: kernel vs mirror on the V4-Flash gate stack.
#[test]
#[ignore = "hw: needs the box, the V4-Flash shard and $BLOOMERY_DATA/ref"]
fn hw_iq3xxs_kernel_matches_mirror() {
    V4Case::load(GgmlType::IQ3_XXS, ROWS).gate_a();
}

/// IQ3_XXS gate B: within 1 ULP of ik's `mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1>`.
#[test]
#[ignore = "hw: needs the box, the V4-Flash shard and $BLOOMERY_DATA/ref"]
fn hw_iq3xxs_kernel_predicts_ik() {
    V4Case::load(GgmlType::IQ3_XXS, 64).gate_b(
        "iq3xxs-ik-dot.txt",
        "mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1>",
    );
}

/// MXFP4 gate 0: `quantize_col` codes the column byte for byte as ik's `quantize_row_q8_2_x4`.
#[test]
#[ignore = "hw: needs the box, the V4-Flash shard and $BLOOMERY_DATA/ref"]
fn hw_mxfp4_encoder_matches_ik() {
    let c = V4Case::load(GgmlType::MXFP4, 1);
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

/// MXFP4 gate A: kernel vs mirror on the V4-Flash down stack.
#[test]
#[ignore = "hw: needs the box, the V4-Flash shard and $BLOOMERY_DATA/ref"]
fn hw_mxfp4_kernel_matches_mirror() {
    V4Case::load(GgmlType::MXFP4, ROWS).gate_a();
}

/// MXFP4 gate B: within 1 ULP of ik's `mul_mat_qX_1_q8_2_T<MXFP4_Unpacker, 1>`.
#[test]
#[ignore = "hw: needs the box, the V4-Flash shard and $BLOOMERY_DATA/ref"]
fn hw_mxfp4_kernel_predicts_ik() {
    V4Case::load(GgmlType::MXFP4, 64).gate_b(
        "mxfp4-x4-ik-dot.txt",
        "mul_mat_qX_1_q8_2_T<MXFP4_Unpacker, 1>",
    );
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
    fn code(&mut self) -> i8 {
        ((self.next() % 255) as i32 - 127) as i8
    }
}

/// Weight f16 scales and activation f32 scales.
const W_SCALES: [u16; 6] = [0x3C00, 0x3800, 0x4000, 0x3B00, 0x0040, 0xBC00];
const A_SCALES: [f32; 6] = [1.0, 0.5, 2.0, 0.00390625, 0.25, -1.0];

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
    let Some(got) = qdot::dot_f32(&bytes, &x) else {
        return; // no AVX2+FMA: the caller's scalar loop owns the value
    };
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

/// The gates below only mean something where the AVX2 encoder is running.
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
        for (name, x) in [
            ("all_nan", &all_nan),
            ("last_in_lane", &last_in_lane),
            ("dropped_max", &dropped_max),
            ("inf", &inf),
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
/// over all 65,536 bit patterns, the rest fixed pseudo-random bytes.
#[test]
fn f16_scales_bit_identical_kernel_vs_mirror() {
    let mut rng = Lcg(0x00F1_6C5C_A1E5);
    // (type, block bytes, offsets of its f16 scale fields)
    for (w, block, fields) in [
        (GgmlType::Q3_K, 110usize, &[108usize][..]),
        (GgmlType::Q4_K, 144, &[0, 2][..]),
        (GgmlType::Q5_K, 176, &[0, 2][..]),
        (GgmlType::Q6_K, 210, &[208][..]),
        (GgmlType::IQ3_XXS, 98, &[0][..]),
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
            for &f in fields {
                row[f..f + 2].copy_from_slice(&h.to_le_bytes());
            }
            let a = dot_row_avx2(w, &row, &acol, 256).unwrap();
            let b = dot_row_scalar(w, &row, &acol, 256).unwrap();
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{w:?} scale {h:#06x}: kernel {a:e} (bits {:#x}) vs mirror {b:e} (bits {:#x})",
                a.to_bits(),
                b.to_bits()
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

/// The q8_2 encoders' bytes for one column, split per block: (codes, d bits,
/// isum), the x4 groups' layout (d at 2i, isum at 8 + 2i, codes at 16 + 32i)
/// then the 36-byte tail blocks'.
fn q82_blocks(col: &[u8], n_blocks: usize) -> Vec<([i8; 32], u16, i16)> {
    const STRIDE: usize = 144;
    const TAIL: usize = 36;
    let nb4 = 4 * (n_blocks / 4);
    let le16 = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let codes = |b: &[u8]| std::array::from_fn(|i| b[i] as i8);
    (0..n_blocks)
        .map(|b| {
            if b < nb4 {
                let (g, ir) = (&col[b / 4 * STRIDE..], b % 4);
                (
                    codes(&g[16 + 32 * ir..]),
                    le16(&g[2 * ir..]),
                    le16(&g[8 + 2 * ir..]) as i16,
                )
            } else {
                let tb = &col[nb4 / 4 * STRIDE + (b - nb4) * TAIL..];
                (codes(&tb[4..]), le16(tb), le16(&tb[2..]) as i16)
            }
        })
        .collect()
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
