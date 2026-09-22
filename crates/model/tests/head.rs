//! Gate for the output head: `l_out-26` in, `result_norm` then `result_output` out.
//!
//! The reference graph computes the lm_head for the last position only: `result_norm`
//! and `result_output` are `{*, 1}` in the manifest — and so is `l_out-26` itself
//! (blocks 24/25 are `{2048, 6}`; the spec's premise of a `{2048, 6}` input is wrong,
//! corrected in the round report). The batch contract is therefore gated in two parts:
//! the oracle's single-column truth in `hw_head_matches_oracle`, and column
//! independence of a 6-wide input in `hw_head_batch_is_column_independent`.
//!
//! The reference's own activation dialect for Q6_K is ik's Q8_2_X4 (an AVX2-only
//! fork trait) — identified and matched inside `matmul_q`; the gate constants
//! below carry the derivation.
//!
//! `hw_` prefix: needs the box, the model file and `$BLOOMERY_DATA/ref`.
#[path = "common/oracle.rs"]
mod oracle;

use model::head::head;
use model::ops::{Tensor2, f32_tensor, rms_norm};

/// Logit gate: `|got - want| <= ATOL + RTOL * |want|`.
///
/// The reference logits run ik's fork-specific Q6_K path: on AVX2 its ggml.c
/// type traits set `Q6_K.vec_dot_type = Q8_2_X4`, so the activation row is
/// quantized per 32 elements with a bf16-rounded scale (`d = bf16rt(amax/127)`,
/// codes `round_even(v * (1/d))`, iqk_quantize.cpp ~1089) and dotted in exact
/// integer math per 16-element group (iqk_gemm_kquants.cpp ~948).
/// `matmul_q` sends Q6_K down the fused kernel: `qdot::quantize_col` writes that
/// same Q8_2_X4 dialect and `dot_q6k_q82x4_avx2` is the port of the dot above, so
/// these constants hold the end-to-end residual of that match (this gate's own
/// `rms_norm` input error plus the elementwise `d*q` reconstruction). They were
/// derived with ~5x headroom on the worst entry against the scalar
/// `quantize_row_q8_2_x4_roundtrip` encoder, which is the path a type with no
/// fused kernel still takes. A stock
/// Q8_K-for-everything `matmul_q` fails them by ~400x, which is the bug class
/// the relative form exists to catch.
const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-3;

/// Relative-scale comparison in `assert_close`'s reporting style: WHERE it deviates,
/// not just that it does. This is the head round's own tool; the shared `assert_close`
/// in `common/oracle.rs` stays untouched.
fn assert_close_rel(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs reference {}",
        got.len(),
        want.len()
    );
    let mut worst_abs = 0.0f32;
    let mut max_abs_ref = 0.0f32;
    let mut worst_rel = 0.0f32; // |diff| / |want|, reported only where |want| >= 1
    let mut usage = 0.0f32; // |diff| / (ATOL + RTOL*|want|) at the worst entry
    let mut at = 0usize;
    let mut pair = (0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        worst_abs = worst_abs.max(d);
        max_abs_ref = max_abs_ref.max(w.abs());
        if w.abs() >= 1.0 {
            worst_rel = worst_rel.max(d / w.abs());
        }
        let u = d / (ATOL + RTOL * w.abs());
        if u > usage {
            usage = u;
            at = i;
            pair = (g, w);
        }
    }
    assert!(
        usage <= 1.0,
        "{what}: max |diff| = {} at index {at} (got {}, reference {}); \
         gate {ATOL:e} + {RTOL:e}*|ref| exceeded {usage:.2}x",
        (pair.0 - pair.1).abs(),
        pair.0,
        pair.1
    );
    eprintln!(
        "{what:38} max|diff| = {worst_abs:e}   max|ref| = {max_abs_ref:e}   \
         worst rel (|ref|>=1) = {worst_rel:.3e}   ok (gate {ATOL:e} + {RTOL:e}*|ref|)"
    );
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(5);
    idx.into_iter().map(|i| (i, v[i])).collect()
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_head_matches_oracle() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

    // The manifest pins the sequence; any other token set invalidates every number below.
    assert_eq!(
        o.tokens,
        [100_000, 549, 6077, 280, 7239, 317],
        "reference values are for the fixed six-token sequence"
    );

    // The oracle's l_out-26 is last-position only ({2048, 1}), occurrence 0, the
    // residual ADD — see the module doc.
    let (xs, xinf) = o.load("l_out-26", 0);
    assert_eq!(xinf.op, "ADD", "l_out-26 must be the residual add");
    assert_eq!([xinf.ne[0], xinf.ne[1]], [2048, 1]);
    let x = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);

    let vocab = g
        .arch_get_u64("vocab_size")
        .expect("vocab_size must come from the file, never a literal") as usize;

    let got = head(&g, &x).unwrap();

    // Shape first: [vocab, n_tokens] — width against the file's own metadata, token
    // count carried through from the input.
    let (want, rinf) = o.load("result_output", 0);
    assert_eq!(
        got.ne0, vocab,
        "logits width must equal the file's vocab_size"
    );
    assert_eq!(
        [got.ne0 as i64, got.ne1 as i64],
        [rinf.ne[0], rinf.ne[1]],
        "shape must match the reference before the values can mean anything"
    );

    // Stage 1, `result_norm` at 1e-4: recomputed here with this round's weights
    // (output_norm.weight, not the ops gate's blk.0.attn_norm.weight). The eps is
    // read independently of head's own lookup — same key, second reader.
    let (norm_want, ninf) = o.load("result_norm", 0);
    let eps = g
        .value("deepseek2.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.as_f32())
        .expect("rms eps must come from the file, never from a literal");
    let norm_t = g.find("output_norm.weight").unwrap();
    let gain = f32_tensor(&g, norm_t).unwrap();
    let normed = rms_norm(&x, &gain, eps);
    assert_eq!(
        [normed.ne0 as i64, normed.ne1 as i64],
        [ninf.ne[0], ninf.ne[1]]
    );
    oracle::assert_close(
        &normed.data,
        &norm_want,
        1e-4,
        "head stage 1 -> result_norm",
    );

    // Stage 2, the logits, at logit scale — see RTOL/ATOL for why relative.
    assert_close_rel(&got.data, &want, "head -> result_output");

    // The one integer round 1-5 ultimately compares: same top token, exactly.
    assert_eq!(
        argmax(&got.data),
        argmax(&want),
        "argmax token must match the reference exactly"
    );
    let id = argmax(&got.data);
    eprintln!("argmax token id: {id}");
    eprintln!("top-5 got: {:?}", top5(&got.data));
    eprintln!("top-5 ref: {:?}", top5(&want));
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_head_batch_is_column_independent() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

    let (xs, xinf) = o.load("l_out-26", 0);
    let ne0 = xinf.ne[0] as usize;
    let n = 6;
    let vocab = g
        .arch_get_u64("vocab_size")
        .expect("vocab_size must come from the file, never a literal") as usize;

    // A 6-wide input: columns 0..4 are distinct scaled variants (so cross-column
    // contamination cannot cancel out), column 5 is the oracle's input verbatim.
    let scales = [0.5f32, 0.75, 1.25, 1.5, 2.0, 1.0];
    let mut wide = Tensor2::zeros(ne0, n);
    for (c, &s) in scales.iter().enumerate() {
        for (dst, &v) in wide.col_mut(c).iter_mut().zip(&xs) {
            *dst = v * s;
        }
    }

    let wide_out = head(&g, &wide).unwrap();
    assert_eq!(
        (wide_out.ne0, wide_out.ne1),
        (vocab, n),
        "a batch in must give a batch out"
    );

    // The last column is the oracle-gated truth again, at the same gate.
    let (want, _) = o.load("result_output", 0);
    assert_close_rel(
        wide_out.col(n - 1),
        &want,
        "head(6-wide) last column -> result_output",
    );

    // Every column must equal the same column computed alone — bitwise, because
    // rms_norm, the Q8_K row quantization and the dot are all per-column.
    for c in 0..n {
        let alone = head(&g, &Tensor2::from_vec(ne0, 1, wide.col(c).to_vec())).unwrap();
        assert_eq!(
            alone.data,
            wide_out.col(c),
            "column {c} must not depend on the other columns"
        );
    }
    eprintln!(
        "columns 0..4 of the 6-wide output are self-consistent only; the reference holds \
         no logits for them (ik runs the lm_head on the last position alone)"
    );
}
