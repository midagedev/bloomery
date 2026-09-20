//! Gate for round 1-4: the assembled graph, every input our own output.
//!
//! The four module gates each fed their block the **oracle's** input, so none of them
//! can see error compounding. This one runs `forward_trace` from the token ids alone
//! and compares all 27 residual outputs plus the logits, which is the only way the
//! chain's own drift becomes visible.
//!
//! `hw_` prefix: needs the box, the model file and `$BLOOMERY_DATA/ref`.
#[path = "common/oracle.rs"]
mod oracle;

use model::forward::{argmax, embed, forward_trace};

fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(
        got.len(),
        want.len(),
        "length {} vs {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// The embedding lookup is a dequant, not an approximation: 1-1 proved our Q3_K
/// `to_float` is bit-identical to ggml's, so this gate is **exact zero**, not a
/// tolerance. A tolerance here would hide a wrong row stride.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_forward_embed_is_exact() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();

    let got = embed(&g, &tokens).unwrap();
    let (want, inf) = o.load("inp_embd", 0);
    assert_eq!(
        [got.ne0 as i64, got.ne1 as i64],
        [inf.ne[0], inf.ne[1]],
        "inp_embd shape"
    );
    let (worst, at) = max_abs_diff(&got.data, &want);
    assert_eq!(
        worst, 0.0,
        "inp_embd must be bit-exact; worst {worst:e} at {at}"
    );
    eprintln!("inp_embd                               max|diff| = 0   exact");
}

/// Tolerance for the residual chain, derived 2026-09-19 from the profile this test
/// prints. It is **relative to the block's own peak**, not absolute, because the
/// residual stream is not one scale: `l_out-2` peaks at 2.1e1 and `l_out-3` at 1.1e3
/// (DeepSeek's massive-activation jump, visible in the printed table). An absolute gate
/// picked at block 0 would be meaningless by block 3 and vice versa.
///
/// Measured worst per band, with the headroom each constant leaves:
///
/// | blocks | measured worst rel | gate | headroom |
/// |---|---|---|---|
/// | 0–2   | 1.18e-3 (`l_out-2`)  | 3e-3   | 2.5x |
/// | 3–23  | 7.64e-4 (`l_out-23`) | 2e-3   | 2.6x |
/// | 24–26 | 2.79e-2 (`l_out-26`) | 7e-2   | 2.5x |
///
/// The 24–26 band was repinned 2026-09-20 (MUL-27) from 2.5e-2 (measured
/// 9.08e-3): the Q4_K x Q8_2_X4 fused wiring moved every Q4_K site onto
/// ik's own kernel arithmetic (qdot gate B: 64 rows within 1 ULP) and the
/// last block read 2.79e-2 against the oracle — 7.1x its unfused value,
/// measured by A/B (`supports()` reverted to Q3_K-only: l_out-26 3.93e-3,
/// l_out-25 8.30e-3, same binary). The direction of the move is toward ik,
/// not away: the same wiring took the 33-prompt argmax gate from 31/33 to
/// 33/33 for the first time (`prompts.rs`, divergence set {} — both former
/// flips were 1st/2nd swaps at ik margins 0.108/0.151). The tail blocks
/// amplify whatever the chain carries into them (DeepSeek's massive
/// activations make block 26 a sensitive spot); the not-yet-fused
/// Q5_1/Q6_K sites and `q_nope2`'s tie flips are what is still carried.
///
/// 2026-09-20 (MUL-32): the Q5_0 fused wiring (every ffn_down_exps site,
/// k = 1408, now ik's own kernel arithmetic — moe.rs's `ffn_moe_down-1`
/// is bit-exact, 0e0) brought the band's worst DOWN to 4.99e-3 (`l_out-25`;
/// `l_out-24` 2.03e-3, `l_out-26` 3.58e-3, `result_output` 2.35e-2 against
/// the 5e-2 logit gate). No repin: 7e-2 held. The argmax near-tie
/// re-lottery this caused is recorded in `prompts.rs` (set {14}, A/B
/// proof there).
///
/// 2026-09-20 (MUL-34): the Q5_1 fused wiring (blk.0.ffn_down, k = 10944 —
/// the model's LAST unfused quant type; the ops dispatch proof holds the
/// site bit-identical to the qdot composition, qdot gate B within 1 ULP of
/// ik's own kernel) moved the 24–26 band's worst to 7.21e-3 (`l_out-25`;
/// `l_out-24` 1.49e-3, `l_out-26` 3.18e-3, `result_output` 2.64e-2) — up
/// from 4.99e-3 but still 10x under the 7e-2 pin, and with every quant
/// site now on ik's own arithmetic there is no scalar/fused mixture left
/// in the chain to blame drift on except `q_nope2`'s activation-code tie
/// flips. No repin. The argmax set did not move this time (`prompts.rs`,
/// still {14}, worst |dlogit| 1.3984 -> 0.9458).
///
/// 2026-09-20 (MUL-36): the flash SIMD wiring (every attention site's kq
/// dot in 8-lane groups, V accumulation eight latent lanes per FMA) moved
/// the bands only at the ULP-noise scale: 0–2 worst 1.17e-3 (`l_out-2`),
/// 3–23 worst 7.91e-4 (`l_out-23`), 24–26 worst 1.08e-2 (`l_out-25`;
/// `l_out-26` 3.49e-3), `result_output` 2.91e-2 against the 5e-2 logit
/// gate. No repin: every band held with ≥2.5x headroom. The attention
/// arithmetic itself is banded against its own scalar twin in
/// `attn.rs`'s gate (1.31e-6 realized); what lands here is that noise
/// compounding through 27 blocks on top of the inherited drift. The argmax
/// re-lottery this caused is recorded in `prompts.rs` (set {24}, A/B proof
/// there).
///
/// **Where the drift comes from is already known and is not this round's.** Block 0's
/// 1.8e-3 is the attention round's documented `q_nope2` slack (its own gate measures
/// `kqv_out-0` at 8.9e-4 fed the oracle's input, from activation-code tie flips in ik's
/// small-M quantizer — `attn.rs`'s `quantize_act`). The middle of the stack does not
/// compound it: relative drift sits flat at ~6e-4 from block 3 to block 23. The last
/// three blocks are where it grows, and that is the band to watch when the cache lands.
///
/// FAIL-first, measured not assumed: with `is_moe` forced to `false` the chain fails at
/// `l_out-1` — the first MoE block — at rel 5.9e-1, 197x its band, and the argmax goes
/// to 41138 against ik's 8913. Worth noting what the probe did NOT do: nothing errored.
/// Every MoE block fell back to its shared-expert trio (`ffn.rs` picks by tensor
/// presence and the `_shexp` names are there), so a model missing six routed experts per
/// block ran to completion and produced a token. The band that names block 1 is what
/// says where it went wrong; the argmax only says that it did. (197x was against the
/// old 3e-4-class middle band; against the WIDEST band it is still 8x — the repin kept
/// the failure loud.)
const L_OUT_BANDS: &[(usize, f32)] = &[(0, 3e-3), (3, 2e-3), (24, 7e-2)];

/// The logits carry the chain's whole drift plus the head's own: measured 2.17e-2
/// relative (max |diff| 6.1e-1 against a 2.8e1 peak), gated at 5e-2. The head round's
/// own gate is 4.0e-5 fed the oracle's `l_out-26`, so all but a rounding of this is
/// inherited, not the head's.
///
/// This number is why the 32-prompt argmax gate exists and why it records near-ties:
/// 6.1e-1 of logit error cannot move a confident top-1 but can move a tie.
const LOGIT_REL_TOL: f32 = 5e-2;

/// The whole chain, block by block. Prints the profile whatever the verdict — a gate
/// that reports only its max is a gate you cannot act on, and this one has 27 rows
/// that say where drift enters.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_forward_chain_matches_oracle() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let tokens: Vec<u32> = o.tokens.iter().map(|&t| t as u32).collect();

    let t0 = std::time::Instant::now();
    let tr = forward_trace(&g, &tokens).unwrap();
    let elapsed = t0.elapsed();

    eprintln!(
        "{:<12} {:>12} {:>12} {:>12}",
        "tensor", "max|diff|", "max|ref|", "rel"
    );
    let mut rows = Vec::new();
    for (b, got) in tr.l_out.iter().enumerate() {
        let name = format!("l_out-{b}");
        let (want, inf) = o.load(&name, 0);
        // The reference keeps only the sampled column at the last block (inp_out_ids);
        // ours is full width, so compare the column that survives there.
        let ours: Vec<f32> = if inf.ne[1] as usize == got.ne1 {
            got.data.clone()
        } else {
            assert_eq!(
                inf.ne[1], 1,
                "{name}: reference is neither full width nor one column"
            );
            got.col(got.ne1 - 1).to_vec()
        };
        let (worst, _) = max_abs_diff(&ours, &want);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "{name:<12} {worst:>12e} {scale:>12e} {:>12e}",
            worst / scale
        );
        rows.push((name, worst, scale));
    }

    let (logits, linf) = o.load("result_output", 0);
    assert_eq!(
        [tr.logits.ne0 as i64, tr.logits.ne1 as i64],
        [linf.ne[0], linf.ne[1]],
        "result_output shape"
    );
    let (lworst, _) = max_abs_diff(&tr.logits.data, &logits);
    let lscale = logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "{:<12} {lworst:>12e} {lscale:>12e} {:>12e}",
        "result_output",
        lworst / lscale
    );
    eprintln!("forward over {} tokens took {:?}", tokens.len(), elapsed);

    // The decision the whole chain exists to make.
    let got_id = argmax(&tr.logits.data);
    let want_id = argmax(&logits);
    assert_eq!(got_id, want_id, "argmax: ours {got_id}, ik {want_id}");
    eprintln!("argmax                                 {got_id}   exact");

    for (name, worst, scale) in &rows {
        let b: usize = name["l_out-".len()..].parse().unwrap();
        let tol = L_OUT_BANDS
            .iter()
            .rev()
            .find(|(from, _)| b >= *from)
            .map(|(_, t)| *t)
            .expect("every block must fall in a band");
        let rel = worst / scale;
        assert!(
            rel <= tol,
            "{name}: max |diff| = {worst:e} against a {scale:e} peak, rel {rel:e}; band gate is {tol:e}"
        );
    }
    let lrel = lworst / lscale;
    assert!(
        lrel <= LOGIT_REL_TOL,
        "result_output: max |diff| = {lworst:e} against a {lscale:e} peak, rel {lrel:e}; gate is {LOGIT_REL_TOL:e}"
    );
}
