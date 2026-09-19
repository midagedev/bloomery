//! Gate for `model::ffn`: the dense FFN (block 0) and the shared expert (block 1)
//! against the oracle's staged tensors.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
#[path = "common/oracle.rs"]
mod oracle;

use model::ffn::{dense_ffn, dense_ffn_up_gate};
use model::ops::Tensor2;

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}
// NOTE: `model_path` duplicates `tests/ops.rs` on purpose — integration tests are
// separate crates and can only share via `tests/common/`, which this round does
// not own (file whitelist: this file and `src/ffn.rs`).

fn load_input(o: &oracle::Oracle, name: &str) -> Tensor2 {
    let (data, info) = o.load(name, 0);
    Tensor2::from_vec(info.ne[0] as usize, info.ne[1] as usize, data)
}

/// The manifest's token line is what distinguishes the 6-token graph (1155
/// tensors) from the 1-token graph (1101) — a gate that skips this can compare
/// the right names from the wrong graph while staying green (`docs/oracle.md`).
fn check_tokens(o: &oracle::Oracle) {
    assert_eq!(
        o.tokens,
        vec![100000, 549, 6077, 280, 7239, 317],
        "oracle token sequence must be the fixed six-token run"
    );
    eprintln!("oracle model: {}", o.model);
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_dense_ffn_block0_matches_ggml() {
    let o = oracle::Oracle::open();
    check_tokens(&o);
    let g = gguf::Gguf::open(model_path()).unwrap();

    // The intermediate width is the file's, not a literal (cf. the ops gate,
    // where eps comes from the file and never from a literal).
    let ff_len = g
        .arch_get_u64("feed_forward_length")
        .expect("deepseek2.feed_forward_length");
    assert_eq!(ff_len, 10944, "spec premise: dense intermediate width");

    let x = load_input(&o, "ffn_norm-0");

    let up = dense_ffn_up_gate(&g, 0, &x).unwrap();
    let (want_up, up_inf) = o.load("ffn_up_gate-0", 0);
    assert_eq!(
        [up.ne0 as i64, up.ne1 as i64],
        [up_inf.ne[0], up_inf.ne[1]],
        "shape must match the reference before the values can mean anything"
    );
    assert_eq!(
        up_inf.op, "FUSED_UP_GATE",
        "ffn_up_gate-0 must be the fused gate/up product"
    );
    assert_eq!(up.ne0 as u64, ff_len);
    oracle::assert_close(
        &up.data,
        &want_up,
        1e-4,
        "dense_ffn_up_gate -> ffn_up_gate-0",
    );

    let out = dense_ffn(&g, 0, &x).unwrap();
    let (want_out, out_inf) = o.load("ffn_out-0", 0);
    assert_eq!(
        [out.ne0 as i64, out.ne1 as i64],
        [out_inf.ne[0], out_inf.ne[1]],
        "shape must match the reference before the values can mean anything"
    );
    assert_eq!(
        out_inf.op, "MUL_MAT",
        "ffn_out-0 must be the down projection"
    );
    oracle::assert_close(&out.data, &want_out, 1e-4, "dense_ffn -> ffn_out-0");
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_dense_ffn_shexp_block1_matches_ggml() {
    let o = oracle::Oracle::open();
    check_tokens(&o);
    let g = gguf::Gguf::open(model_path()).unwrap();

    // 2 shared experts × 1408, from the file's own keys.
    let shared = g
        .arch_get_u64("expert_shared_count")
        .expect("deepseek2.expert_shared_count");
    let expert_ff = g
        .arch_get_u64("expert_feed_forward_length")
        .expect("deepseek2.expert_feed_forward_length");
    assert_eq!(
        (shared, expert_ff),
        (2, 1408),
        "spec premise: shared-expert geometry"
    );

    let x = load_input(&o, "ffn_norm-1");

    // Same function as block 0, no special case: the trio resolves by presence.
    let up = dense_ffn_up_gate(&g, 1, &x).unwrap();
    let (want_up, up_inf) = o.load("ffn_up_gate-1", 0);
    assert_eq!(
        [up.ne0 as i64, up.ne1 as i64],
        [up_inf.ne[0], up_inf.ne[1]],
        "shape must match the reference before the values can mean anything"
    );
    assert_eq!(
        up_inf.op, "FUSED_UP_GATE",
        "ffn_up_gate-1 must be the fused gate/up product"
    );
    assert_eq!(up.ne0 as u64, shared * expert_ff);
    oracle::assert_close(
        &up.data,
        &want_up,
        1e-4,
        "dense_ffn_up_gate shexp -> ffn_up_gate-1",
    );

    let out = dense_ffn(&g, 1, &x).unwrap();
    let (want_out, out_inf) = o.load("ffn_shexp-1", 0);
    assert_eq!(
        [out.ne0 as i64, out.ne1 as i64],
        [out_inf.ne[0], out_inf.ne[1]],
        "shape must match the reference before the values can mean anything"
    );
    assert_eq!(
        out_inf.op, "MUL_MAT",
        "ffn_shexp-1 must be the down projection"
    );
    oracle::assert_close(&out.data, &want_out, 1e-4, "dense_ffn shexp -> ffn_shexp-1");
}
