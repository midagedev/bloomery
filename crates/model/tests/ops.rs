//! Lead-owned gate for the two primitives every other module stands on. If these are wrong,
//! four rounds fail for a reason that is not theirs.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
#[path = "common/oracle.rs"]
mod oracle;

use model::ops::{Tensor2, matmul_q, rms_norm};

fn model_path() -> String {
    std::env::var("MULLE_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

#[test]
#[ignore = "hw: needs the box, the model file and $MULLE_DATA/ref"]
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
#[ignore = "hw: needs the box, the model file and $MULLE_DATA/ref"]
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
    assert_eq!(winf.op, "MUL_MAT", "occurrence 0 of q-0 must be the matmul, not the concat");
    // 1e-4 on values that reach 18: the residual is f32 accumulation order against ggml's
    // integer sum, measured at 1.5e-5. Before activations went through Q8_K it was 1.2e-1.
    oracle::assert_close(&got.data, &want, 1e-4, "matmul_q -> q-0");
}
