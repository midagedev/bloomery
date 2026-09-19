//! Lead-owned gate for the two primitives every other module stands on. If these are wrong,
//! four rounds fail for a reason that is not theirs.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
#[path = "common/oracle.rs"]
mod oracle;

use gguf::GgmlType;
use model::ops::{Tensor2, matmul_q, rms_norm};

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_rms_norm_matches_ggml() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();

    let (inp, inf) = o.load("inp_embd", 0);
    let x = Tensor2::from_vec(inf.ne[0] as usize, inf.ne[1] as usize, inp);

    let gain_t = g.find("blk.0.attn_norm.weight").unwrap();
    let gain_bytes = g.data(gain_t).unwrap();
    let gain: Vec<f32> = gain_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let eps = g
        .value("deepseek2.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.as_f32())
        .expect("rms eps must come from the file, never from a literal");

    let got = rms_norm(&x, &gain, eps);
    let (want, winf) = o.load("attn_norm-0", 0);
    assert_eq!(
        [got.ne0 as i64, got.ne1 as i64],
        [winf.ne[0], winf.ne[1]],
        "shape must match the reference before the values can mean anything"
    );
    oracle::assert_close(&got.data, &want, 1e-4, "rms_norm -> attn_norm-0");
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_matmul_q_matches_ggml() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();

    let (xs, xinf) = o.load("attn_norm-0", 0);
    let x = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);

    let w = g.find("blk.0.attn_q.weight").unwrap();
    let got = matmul_q(&g, w, &x).unwrap();

    // q-0 occurs twice in the graph: MUL_MAT {3072, 6} then CONCAT {576, 6, 16}.
    // Occurrence 0 is the one this produces; asking by name alone would compare the wrong one.
    let (want, winf) = o.load("q-0", 0);
    assert_eq!([got.ne0 as i64, got.ne1 as i64], [winf.ne[0], winf.ne[1]]);
    assert_eq!(
        winf.op, "MUL_MAT",
        "occurrence 0 of q-0 must be the matmul, not the concat"
    );
    // 1e-4 on values that reach 18: the residual is f32 accumulation order against ggml's
    // integer sum, measured at 1.5e-5. Before activations went through Q8_K it was 1.2e-1.
    oracle::assert_close(&got.data, &want, 1e-4, "matmul_q -> q-0");
}

/// The dispatch proof for the qdot wiring (MUL-21, 2026-09-20): `matmul_q` must route
/// Q3_K with k a multiple of 256 through the fused kernel and leave every other type
/// on the scalar path. The two paths are known NOT to agree bit for bit — the fused
/// kernel is the more accurate one (measured against an f64 exact answer in the qdot
/// round) — so this test proves the move HAPPENED, not that it did not: bit equality
/// against the fused composition (ground A), at least one differing element against
/// the old scalar composition (ground B), and bit equality of a non-Q3_K matmul
/// against the scalar composition (leak watch).
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_matmul_q_q3k_fused_dispatch() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();

    // Same input as hw_matmul_q_matches_ggml: attn_norm-0 through blk.0.attn_q.
    let (xs, xinf) = o.load("attn_norm-0", 0);
    let x = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);
    let w = g.find("blk.0.attn_q.weight").unwrap();
    assert_eq!(w.ty, GgmlType::Q3_K, "this test is the Q3_K dispatch proof");
    let k = w.dims[0] as usize;
    let n = w.dims[1] as usize;
    assert!(
        k.is_multiple_of(256),
        "dispatch needs k % 256 == 0, k = {k}"
    );
    let got = matmul_q(&g, w, &x).unwrap();

    let bytes = g.data(w).unwrap();
    let row_bytes = bytes.len() / n;

    // Ground A — the fused composition assembled here by hand: every column
    // quantized with qdot, every row dotted with qdot. matmul_q must be bit for
    // bit this, or the fused path never ran.
    let cb = qdot::col_bytes(GgmlType::Q3_K, k);
    let mut acol = vec![0u8; x.ne1 * cb];
    for t in 0..x.ne1 {
        qdot::quantize_col(GgmlType::Q3_K, x.col(t), &mut acol[t * cb..(t + 1) * cb]);
    }
    for r in 0..n {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for t in 0..x.ne1 {
            let v = qdot::dot_row(GgmlType::Q3_K, src, &acol[t * cb..(t + 1) * cb], k).unwrap();
            assert_eq!(
                got.data[t * n + r].to_bits(),
                v.to_bits(),
                "row {r} token {t}: must be bit-identical to the fused composition"
            );
        }
    }
    eprintln!(
        "ground A: {n} rows x {} tokens bit-identical to the qdot composition",
        x.ne1
    );

    // Ground B — the OLD scalar composition: dequant_row per weight row,
    // quantize_activations round trip per column, f32 dot ascending. Must differ
    // somewhere: if it matched everywhere the fused path never ran and ground A
    // above passed by coincidence.
    let cols: Vec<Vec<f32>> = (0..x.ne1)
        .map(|t| {
            let mut q = vec![0.0f32; k];
            gguf::quantize_activations(GgmlType::Q3_K, x.col(t), &mut q);
            q
        })
        .collect();
    let mut row = vec![0.0f32; k];
    let mut diffs = 0usize;
    let mut first: Option<(usize, usize)> = None;
    for r in 0..n {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        gguf::dequant_row(GgmlType::Q3_K, src, &mut row).unwrap();
        for t in 0..x.ne1 {
            let xc = &cols[t];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += row[i] * xc[i];
            }
            if acc.to_bits() != got.data[t * n + r].to_bits() {
                diffs += 1;
                first.get_or_insert((r, t));
            }
        }
    }
    assert!(
        diffs >= 1,
        "bit-identical to the scalar composition — the fused path did not run, \
         so ground A proved nothing ({} elements compared)",
        n * x.ne1
    );
    let (r0, t0) = first.unwrap();
    eprintln!(
        "ground B: {diffs} of {} elements differ from the scalar composition \
         (first at row {r0} token {t0}) — Q3_K moved onto the fused path",
        n * x.ne1
    );

    // Leak watch — a non-Q3_K matmul must stay on the scalar path, bit for bit.
    // blk.0.ffn_down is Q5_1; the input is synthetic (all ones, k-matched), so
    // this needs no oracle entry. Any difference here means the Q3_K wiring
    // reached into another type's path.
    let wf = g.find("blk.0.ffn_down.weight").unwrap();
    assert_eq!(wf.ty, GgmlType::Q5_1, "the leak watch is a Q5_1 statement");
    let kf = wf.dims[0] as usize;
    let nf = wf.dims[1] as usize;
    let xf = Tensor2::from_vec(kf, 1, vec![1.0f32; kf]);
    let gotf = matmul_q(&g, wf, &xf).unwrap();
    let fbytes = g.data(wf).unwrap();
    let frow_bytes = fbytes.len() / nf;
    let mut fq = vec![0.0f32; kf];
    gguf::quantize_activations(GgmlType::Q5_1, xf.col(0), &mut fq);
    let mut frow = vec![0.0f32; kf];
    for r in 0..nf {
        let src = &fbytes[r * frow_bytes..(r + 1) * frow_bytes];
        gguf::dequant_row(GgmlType::Q5_1, src, &mut frow).unwrap();
        let mut acc = 0.0f32;
        for i in 0..kf {
            acc += frow[i] * fq[i];
        }
        assert_eq!(
            gotf.data[r].to_bits(),
            acc.to_bits(),
            "row {r}: the Q3_K wiring must not touch the Q5_1 scalar path"
        );
    }
    eprintln!("leak watch: Q5_1 all-ones x ffn_down bit-identical over {nf} rows");
}
