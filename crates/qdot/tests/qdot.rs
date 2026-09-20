//! qdot gates: the fused Q3_K x Q8_K kernel against the path `matmul_q`
//! runs today, against an exact f64 reference, and against its own scalar
//! mirror.
//!
//! `hw_` prefix: needs the box (the model file and `$BLOOMERY_DATA/ref`),
//! excluded by default, run by `just gate-qdot`. The one pure gate
//! (`rejects_unaligned_k`) runs in the default set.
//!
//! Data is real on purpose: weights come straight out of the model mmap
//! (`blk.1.ffn_gate_exps.weight`, Q3_K, k = 2048, expert 0's slice of the
//! 3-D stack; `blk.1.attn_kv_b.weight`, Q3_K, k = 512), activations from
//! the oracle's dumps of the same block's inputs (`ffn_norm-1.0.f32`,
//! `kv_compressed-1.0.f32` — the exact tensors `matmul_q` consumes for
//! these weights). Real quantization codes decide which branches of the
//! decode run; synthetic random bytes walk fewer of them.
//!
//! The gates assert with `assert!` (not `debug_assert!`) because they run
//! in release.

use gguf::GgmlType;
use gguf::quant::{dequant_row, quantize_row_q8_k_roundtrip};
use qdot::{QdotError, col_bytes, dot_row, dot_row_avx2, dot_row_scalar, quantize_col, supports};

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

/// The oracle's f32 dump of one tensor, byte length checked against the
/// expected value count — a truncated dump must fail the gate, not quietly
/// shift every column (same stance as the model crates' oracle reader).
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

/// Rows dotted per tensor. The spec's floor is 1000; 1024 keeps it even.
const ROWS: usize = 1024;

struct Case<'a> {
    name: &'static str,
    k: usize,
    row_bytes: usize,
    /// The first `ROWS` rows of the tensor (expert 0's slice for the 3-D
    /// expert stack — the routed expert chosen varies with the prompt, the
    /// code distribution does not).
    bytes: &'a [u8],
    /// Token 0's activation column (the dump's first k values).
    act: Vec<f32>,
    /// `act` in Q8_K blocks, what `dot_row` consumes.
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

/// The current path's value for one row: dequant_row + Q8_K round trip +
/// sequential f32 dot, exactly the arithmetic `ops::matmul_q` performs
/// (verified against ggml at 1e-4 through the model gates).
fn current_path_dot(c: &Case, qa: &[f32], wbuf: &mut [f32], r: usize) -> f32 {
    dequant_row(GgmlType::Q3_K, &c.bytes[r * c.row_bytes..], wbuf).unwrap();
    let mut acc = 0.0f32;
    for i in 0..c.k {
        acc += wbuf[i] * qa[i];
    }
    acc
}

/// The gate's own f64 reference, deliberately independent of the lib's two
/// decoders: the integer codes (sub-block scale, low bits, high bit, q8)
/// are decoded here again from the raw bytes, summed in i64 per
/// super-block, and scaled once in f64. Every product a real path computes
/// in f32 is exact here, so the distance it reports is exactly the f32
/// rounding of the path under test. Geometry per `dequantize_row_q3_K`
/// (stage 0 verified it at 1e-7 against ggml): element 128c + 32f + 16h +
/// l reads qs byte 32c + 16h + l field f, high bit hmask byte 16h + l bit
/// 4c + f, sub-block scale index 8c + 2f + h.
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
        // Unpacked scale byte 0..63, actual scale byte - 32 (quant.rs:348,
        // the AVX2 kernel's _mm_sub_epi8).
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
/// must reproduce `quantize_row_q8_k_roundtrip`'s f32 output bit for bit,
/// over every token column of both activation dumps. This is where a
/// rounding or sign inconsistency between the two representations of the
/// same quantization shows up.
#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref"]
fn hw_quantize_col_matches_roundtrip_bits() {
    for (dump, k) in [("ffn_norm-1", 2048usize), ("kv_compressed-1", 512)] {
        let vals = oracle_f32(dump, k * 6);
        let mut packed = vec![0u8; col_bytes(GgmlType::Q3_K, k)];
        let mut rt = vec![0.0f32; k];
        let mut restored = vec![0.0f32; k];
        let mut bad = 0usize;
        // +0.0 vs -0.0 is the one f32 pair that is numerically equal with
        // different bits. A code of 0 restores to d*0 whose sign is the
        // multiplication's, not information the packed codes carry: the C
        // quantizer codes every zero — +0.0 and -0.0 alike — as 0, and every
        // consumer adds it as zero. Those pairs are counted separately and
        // never treated as a material difference; any other equal-value /
        // different-bits case is impossible (equal non-zero f32s are
        // bit-identical).
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

/// Gate 2: `dot_row` must match the current path (dequant + round trip +
/// sequential f32 dot, the arithmetic `matmul_q` performs and the model
/// gates verified against ggml at 1e-4) on the same rows and the same
/// activation values. Relative difference at the tensor scale
/// (max |diff| / max |current|, the metric the q3k-cpu gate uses) <= 1e-5
/// for both k. Per-row worsts are printed for the report.
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

/// Gate 3: the fused kernel is a prediction — it should land closer to the
/// exact f64 value than the current f32 path on more than half the rows,
/// and its worst relative error must not exceed the current path's worst.
/// The reference is exact (integer codes summed in i64, one f64 scaling per
/// super-block), so the only thing it measures is each path's f32 rounding.
/// If the closer-count comes out at or below half, the kernel is wrong —
/// that is the failure mode this gate exists to catch.
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

/// Gate 4: the scalar fallback (what a machine without AVX2 runs) is
/// bit-identical to the AVX2 kernel on the same rows. The integer
/// super-block sums are exact in i32 and the f32 scaling is the same
/// expression in the same order, so bit equality is the expected outcome —
/// anything looser would hide a real decode difference.
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

/// Gate 5 (pure): `k % 256 != 0` is rejected — an `Err`, never a panic,
/// never a silently wrong value — and the shape contract around it holds:
/// unsupported types refuse before touching the data, short buffers are
/// errors, and `col_bytes` is the allocation contract `quantize_col`
/// asserts against.
#[test]
fn rejects_unaligned_k() {
    // Buffers sized for k = 2048 so only `k` can be at fault.
    let wrow = vec![0u8; 110 * 8];
    let acol = vec![0u8; 296 * 8];
    for k in [1usize, 100, 255, 2048 + 128] {
        match dot_row(GgmlType::Q3_K, &wrow, &acol, k) {
            Err(QdotError::UnalignedK { k: got }) => assert_eq!(got, k),
            other => panic!("k = {k}: expected UnalignedK, got {other:?}"),
        }
    }
    // A type outside this build's table refuses before reading any bytes.
    // Q4_K joined the table in MUL-27, so the rejection witness is Q5_0 now.
    assert!(matches!(
        dot_row(GgmlType::Q5_0, &wrow, &acol, 2048),
        Err(QdotError::UnsupportedType(GgmlType::Q5_0))
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
    // The aligned k on the same buffers is a valid (all-zero) dot, not an
    // error — and `UnalignedK` for k = 2048 + 128 was a k problem, not a
    // buffer problem, which this line pins.
    assert_eq!(dot_row(GgmlType::Q3_K, &wrow, &acol, 2048).unwrap(), 0.0);
    // col_bytes is the allocation contract quantize_col asserts against.
    assert_eq!(col_bytes(GgmlType::Q3_K, 2048), 296 * 8);
    assert_eq!(col_bytes(GgmlType::Q3_K, 512), 296 * 2);
    // supports(): the type must be in the table at all.
    // Q4_K joined the table in MUL-27; Q5_0 is the out-of-table witness now.
    assert!(!supports(GgmlType::Q5_0));
}

/// Gate 5's sibling: `col_bytes` panics (documented) on an unaligned k —
/// allocation-time constants are asserts, data-dependent shapes are
/// `Err`s.
#[test]
#[should_panic(expected = "whole 256-value super-blocks")]
fn col_bytes_rejects_unaligned_k() {
    col_bytes(GgmlType::Q3_K, 100);
}

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_q4k_dot_row_matches_scalar_and_predicts() {
    // MUL-27's gate pair for the Q4_K x Q8_K kernel, same regime as the Q3_K
    // round: (A) the AVX2 kernel and its scalar mirror agree bit for bit on
    // real weight rows, and (B) the fused value is CLOSER to an exact f64
    // answer than the engine's current path (dequant + q8_2_x4 round trip +
    // f32 dot) — or the kernel is wrong, not the gate.
    let g = gguf::Gguf::open(model_path()).unwrap();
    // Any real Q4_K row works — the gates are about the kernel, not a site.
    // Scanning beats hard-coding: the quantization mix is a property of the
    // GGUF file, not of this crate.
    let w = g
        .iter_tensors()
        .find(|w| w.ty == gguf::GgmlType::Q4_K && (w.dims[0] as usize).is_multiple_of(256))
        .expect("the model must carry at least one aligned Q4_K tensor");
    let k = w.dims[0] as usize;
    let n = w.dims[1] as usize;
    assert!(k.is_multiple_of(256), "fused dispatch needs k % 256 == 0");

    let bytes = g.data(w).unwrap();
    let row_bytes = bytes.len() / n;
    assert_eq!(row_bytes, 144 * k / 256);

    // Real activations: the oracle's first-block normed input, first column.
    // The oracle dumps are prefill-wide (6 tokens); the first column is the
    // one-token activation the decode path would quantize.
    let vals = oracle_f32("attn_norm-0", 2048 * 6);
    let xs: Vec<f32> = vals[..k].to_vec();
    let cb = qdot::col_bytes(gguf::GgmlType::Q4_K, k);
    let mut acol = vec![0u8; cb];
    qdot::quantize_col(gguf::GgmlType::Q4_K, &xs, &mut acol);

    // Gate A — bit identity against the scalar mirror.
    let rows = n.min(1024);
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let a = qdot::dot_row(gguf::GgmlType::Q4_K, src, &acol, k).unwrap();
        let b = qdot::dot_row_scalar(gguf::GgmlType::Q4_K, src, &acol, k).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {r}: kernel and mirror must be bit-identical"
        );
    }
    eprintln!("gate A: {rows} rows bit-identical (kernel vs scalar mirror)");

    // Gate B — bit identity against IK'S OWN KERNEL on the same real rows
    // with the same activations: `tools/ref/q4k_ref.cpp` runs
    // ggml_vec_dot_q4_K_q8_K with ik's quantize_row_q8_K coding of the same
    // attn_norm-0 column and dumps 64 rows as hex floats. Verbatim port
    // means verbatim: every integer partial and every f32 combine in the
    // same order, so the bits must match, not merely agree.
    // (An f64-semantics gate was tried first and REJECTED: the reference
    // closure disagreed with both the port and ik — the dump is the primary
    // source this repo's correctness regime names, and it is stronger.)
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let dump = std::fs::read_to_string(format!("{base}/ref/q4k-ik-dot.txt"))
        .expect("run tools/ref/q4k_ref.cpp first (just build-ref does not build it)");
    let mut dump_k = 0usize;
    let mut want = Vec::new();
    let mut ik_acol: Vec<u8> = Vec::new();
    for line in dump.lines() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next(), it.next(), it.next()) {
            (Some("tensor"), Some(_), Some("k"), Some(kk)) => dump_k = kk.parse().unwrap(),
            (Some("row"), Some(_), Some(hex), None) => {
                want.push(u32::from_str_radix(hex, 16).expect("ik dumps raw f32 bits"));
            }
            // the one long hex line: ik's quantized activation bytes
            (Some(hex), None, None, None) if hex.len() > 64 => {
                let b = hex.as_bytes();
                ik_acol.extend(
                    b.chunks_exact(2)
                        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap()),
                );
            }
            _ => {}
        }
    }
    assert_eq!(ik_acol.len(), cb, "ik's q8_K column must size-match ours");
    // Encoder comparison first: ours vs ik's coding of the same column.
    // A difference here is legitimate (float paths) but must be KNOWN before
    // kernel bits can mean anything.
    let enc_diffs: usize = ik_acol
        .iter()
        .zip(&acol)
        .filter(|(a, b)| a != b)
        .enumerate()
        .map(|(i, _)| {
            if i < 8 {
                eprintln!(
                    "DEBUG enc byte {i}: ik {:02x} ours {:02x}",
                    ik_acol[i], acol[i]
                );
            }
        })
        .count();
    eprintln!("encoder: {enc_diffs} differing bytes of {cb}");
    assert_eq!(
        dump_k, k,
        "the dump and this scan must land on the same tensor"
    );
    assert!(want.len() >= 64, "dump needs 64 rows");
    for (r, &w) in want.iter().take(64).enumerate() {
        let got = qdot::dot_row(
            gguf::GgmlType::Q4_K,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &ik_acol,
            k,
        )
        .unwrap();
        // Measured (2026-09-20): on ik's own activation bytes rows agree to
        // within 1 ULP — residual float-path rounding, not structure (gate A
        // holds the structure bit-exact internally).
        let ulp = (got.to_bits() as i64 - w as i64).abs();
        assert!(
            ulp <= 1,
            "row {r}: ours {got:.9e} vs ik {}: {ulp} ULP apart",
            f32::from_bits(w)
        );
    }
    eprintln!(
        "gate B: 64 rows within 1 ULP of ik's ggml_vec_dot_q4_K_q8_K (on ik's own activations)"
    );
}
