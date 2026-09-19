//! Gate for the MoE block: router exact, expert numerics at 1e-4 (the down projection
//! at 4e-4 — dated derivation at its call site), and one structural gate that counts
//! which experts were actually dequantized.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
//! Run on the box: `cd ~/repo/mulle-moe && cargo test -p model --test moe -- --ignored`.
//!
//! Two oracle quirks the router test has to live with (both verified against the files,
//! see the module report):
//!
//! * `ffn_moe_probs-1 (sort)` holds a full 64-entry ranking per token, but only the
//!   first `n_used` entries are meaningful — the tail is the partial-selection
//!   artifact of ggml's top-k, not a real sort order. The gate reads the prefix.
//! * `ffn_moe_topk-1` is a strided VIEW into that argsort output, so its file contains
//!   `sort[0][0..36]`: a valid top-6 only for token 0. It is used as an independent
//!   cross-check of token 0, never for the other tokens.

#[path = "common/oracle.rs"]
mod oracle;

use model::Tensor2;
use model::moe;

fn model_path() -> String {
    std::env::var("MULLE_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

/// `ffn_norm-1` — the MoE block's input, straight from the oracle.
fn moe_input(o: &oracle::Oracle) -> Tensor2 {
    let (xs, xinf) = o.load("ffn_norm-1", 0);
    Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs)
}

#[test]
#[ignore = "hw: needs the box, the model file and $MULLE_DATA/ref"]
fn hw_moe_router_exact() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let x = moe_input(&o);

    let b = moe::route(&g, 1, &x).unwrap();
    let tr = moe::last_trace().expect("route must leave a trace");
    let (n_expert, n_used, n_tokens) = (tr.n_expert, tr.n_used, tr.n_tokens);
    assert_eq!(
        (n_expert, n_used, n_tokens),
        (64, 6, 6),
        "file metadata and oracle batch must agree before ids can mean anything"
    );

    // Routing ids are integers: EXACT match, no tolerance. A wrong expert inside a
    // 1e-4 numeric gate is invisible and is the failure that matters most here.
    let (sort_vals, sort_inf) = o.load("ffn_moe_probs-1 (sort)", 0);
    assert_eq!(
        sort_inf.ty, "i32",
        "the argsort output must be the integer tensor"
    );
    let mut want_ids = Vec::with_capacity(n_used * n_tokens);
    for t in 0..n_tokens {
        want_ids.extend_from_slice(&sort_vals[t * n_expert..t * n_expert + n_used]);
    }
    oracle::assert_exact_i32(&tr.ids, &want_ids, "top-6 ids vs (sort) prefix");

    // Token 0 only: the topk VIEW dump carries token 0's row; the rest is the artifact.
    let (topk_vals, _) = o.load("ffn_moe_topk-1", 0);
    assert_eq!(topk_vals.len(), n_used * n_tokens, "topk view must be 6x6");
    oracle::assert_exact_i32(
        &tr.ids[..n_used],
        &topk_vals[..n_used],
        "top-6 ids token 0 vs topk view",
    );

    // Numerics, in graph order.
    let (logits, linf) = o.load("ffn_moe_logits-1", 0);
    assert_eq!(linf.op, "MUL_MAT");
    oracle::assert_close(&tr.logits, &logits, 1e-4, "route -> ffn_moe_logits-1");

    let (probs, pinf) = o.load("ffn_moe_probs-1", 0);
    assert_eq!(pinf.op, "SOFT_MAX");
    oracle::assert_close(&tr.probs, &probs, 1e-4, "route -> ffn_moe_probs-1");

    let (weights, winf) = o.load("ffn_moe_weights-1", 0);
    assert_eq!(winf.op, "GET_ROWS");
    // Raw softmax probabilities, no renormalization — that is what the reference holds.
    oracle::assert_close(&tr.weights, &weights, 1e-4, "route -> ffn_moe_weights-1");

    // Routing reads the router; it must not have touched any expert stack.
    assert_eq!(
        moe::last_touched_experts(),
        0,
        "route dequantized expert weights"
    );

    // Bucket table sanity: every token lands in exactly n_used buckets, each bucket
    // lists its tokens ascending, and offsets bracket the whole order.
    assert_eq!(b.offsets.len(), n_expert + 1);
    assert_eq!(b.offsets[0], 0);
    assert_eq!(b.offsets[n_expert] as usize, b.order.len());
    assert_eq!(b.order.len(), n_used * n_tokens);
    let mut seen = vec![0u32; n_tokens];
    for e in 0..n_expert {
        let bucket = b.bucket(e);
        assert!(
            bucket.windows(2).all(|w| w[0] < w[1]),
            "bucket {e} not ascending"
        );
        for &t in bucket {
            seen[t as usize] += 1;
        }
    }
    assert!(
        seen.iter().all(|&c| c == n_used as u32),
        "every token must appear in exactly n_used buckets"
    );
}

#[test]
#[ignore = "hw: needs the box, the model file and $MULLE_DATA/ref"]
fn hw_moe_forward_matches_ggml() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let x = moe_input(&o);

    let out = moe::moe_ffn(&g, 1, &x).unwrap();
    let tr = moe::last_trace().expect("moe_ffn must leave a trace");
    assert_eq!((out.ne0, out.ne1), (2048, 6), "ffn_out-1 shape");

    let (par, parinf) = o.load("ffn_moe_gate_par-1", 0);
    assert_eq!(parinf.op, "MOE_FUSED_UP_GATE");
    oracle::assert_close(&tr.gate_par, &par, 1e-4, "moe_ffn -> ffn_moe_gate_par-1");

    let (down, dinf) = o.load("ffn_moe_down-1", 0);
    assert_eq!(dinf.op, "MUL_MAT_ID");
    // This one gate is opened above the spec's 1e-4. The spec's relaxation rule asks
    // for a dated comment, a derivation, and FAIL-first evidence — all three here.
    //
    // 2026-09-19: the reference's fused up/gate kernel accumulates in f32 (AVX2
    // `mul_mat_up_gate_NxM` tiles), so its own `gate_par` — the very tensor
    // `MUL_MAT_ID` then quantizes — sits at most 4.6e-5 from this engine's (p50
    // ~3e-8, measured over all 6*6*1408 entries; this engine accumulates in f64 over
    // operands that are exact in f32, i.e. it holds the truer value). The down
    // quantizer is a discontinuous function of that input (bf16 block scale + integer
    // codes): in the two affected token columns a handful of the 36*44 input blocks
    // sit close enough to a code or bf16-scale boundary that the reference's own noise
    // flips them, and one flipped input code perturbs the whole 2048-entry output
    // column by |w_row| * d16. Measured over this batch: tokens 0/1/2/5 exact to
    // <= 1.4e-6; token 3 has 358 entries over 1e-4 (max 2.81e-4), token 4 has 272
    // (max 2.46e-4) — the discrete signature of ~1-2 flipped codes, not a scale or
    // accumulation error, which would show as a smooth spread across all columns.
    // Cross-check: the same floor appears unchanged when the whole MoE path runs on
    // `ops::matmul_q`'s Q8_K activation quantizer instead of the source-derived
    // q8_2_x4 one (max 2.809167e-4, same per-token distribution) — the floor is a
    // property of the reference, not of the quantizer this engine picks.
    //
    // FAIL-first, both directions (box runs 2026-09-19):
    // * at 1e-4 this comparison fails on the current, correct implementation:
    //   `moe_ffn -> ffn_moe_down-1: max |diff| = 2.809912e-4 at index 46823
    //    (got -0.12958647, reference -0.12986746)`.
    // * at 4e-4 it still catches the real bug class: quantizing the down activations
    //   with the f16 scale format instead of bf16 fails at max |diff| = 2.8e-2,
    //   70x over the gate.
    // 4e-4 is 1.4x the observed maximum and 1/25th of the "1e-2 is a bug" line.
    oracle::assert_close(&tr.down, &down, 4e-4, "moe_ffn -> ffn_moe_down-1");

    // The weighted sum and the shared-expert sum are gated separately: they are two
    // different bugs in the same output tensor (wrong weights vs wrong dense FFN).
    let (routed, rinf) = o.load("ffn_moe_out-1", 0);
    assert_eq!(rinf.op, "MUL_MULTI_ADD");
    oracle::assert_close(&tr.routed_out, &routed, 1e-4, "moe_ffn -> ffn_moe_out-1");

    let (shexp, sinf) = o.load("ffn_shexp-1", 0);
    assert_eq!([sinf.ne[0], sinf.ne[1]], [2048, 6]);
    oracle::assert_close(&tr.shexp_out, &shexp, 1e-4, "moe_ffn -> ffn_shexp-1");

    let (want, winf) = o.load("ffn_out-1", 0);
    assert_eq!(winf.op, "ADD");
    oracle::assert_close(&out.data, &want, 1e-4, "moe_ffn -> ffn_out-1");
}

/// Structure, not numbers: the engine may only dequantize experts some token actually
/// routed to. mistral.rs dequantizes all 64 experts per token on x86_64
/// (`mistralrs-quant/src/gguf/cpu.rs:74`, see docs/research/mistralrs-prior-art.md
/// §4.4) and that failure mode costs 442 ms/layer/token against 0.20 ms. This gate
/// keeps it out of this engine: `moe::last_touched_experts()` is set where the expert
/// bytes are fetched, so an empty-bucket matmul that touches all 64 trips it.
///
/// FAIL-first: verified 2026-09-19 by deleting the empty-bucket skip in `moe_ffn` (the
/// naive version then dequantizes all 64 stacks): it fails with `distinct experts
/// dequantized must equal distinct experts routed to (25), got 64`.
#[test]
#[ignore = "hw: needs the box, the model file and $MULLE_DATA/ref"]
fn hw_moe_touches_only_routed_experts() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(model_path()).unwrap();
    let x = moe_input(&o);

    moe::moe_ffn(&g, 1, &x).unwrap();
    let tr = moe::last_trace().expect("moe_ffn must leave a trace");
    let touched = moe::last_touched_experts();

    let mut routed_mask = 0u64;
    for &e in &tr.ids {
        routed_mask |= 1u64 << e as u64;
    }
    let n_distinct = routed_mask.count_ones() as usize;

    assert_eq!(
        touched.count_ones() as usize,
        n_distinct,
        "distinct experts dequantized must equal distinct experts routed to ({}), got {}",
        n_distinct,
        touched.count_ones()
    );
    assert_eq!(
        touched, routed_mask,
        "dequantized an expert no token routed to — the mistral.rs x86_64 failure mode (prior-art §4.4)"
    );
    assert!(
        n_distinct < tr.n_expert,
        "the reference batch routes to {n_distinct} of {} experts; if it ever routes to all, \
         this gate can no longer distinguish bucketed dispatch from touch-all-64",
        tr.n_expert
    );
}
