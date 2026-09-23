//! Gate for the MoE block: router exact, expert numerics at 1e-4 (the down projection
//! at 4e-4 — dated derivation at its call site), and one structural gate that counts
//! which experts were actually dequantized.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
//! Run on the box: `cd ~/repo/bloomery-moe && cargo test -p model --test moe -- --ignored`.
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
use model::arch::deepseek2::plan;
use model::moe;

/// `ffn_norm-1` — the MoE block's input, straight from the oracle.
fn moe_input(o: &oracle::Oracle) -> Tensor2 {
    let (xs, xinf) = o.load("ffn_norm-1", 0);
    Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs)
}

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_moe_router_exact() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let x = moe_input(&o);
    moe::set_trace_enabled(true);

    let b = plan::route(&g, 1, &x).unwrap();
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
    // PIN(2026-09-21): 1e-4 -> exact. The router is F32 x F32 and `qdot::dot_f32`
    // sums in the reference's lane order, so the logits are the reference's bits;
    // the scalar left-to-right sum it replaced sat 3.8e-6 away and fails this line.
    oracle::assert_close(&tr.logits, &logits, 0.0, "route -> ffn_moe_logits-1");

    let (probs, pinf) = o.load("ffn_moe_probs-1", 0);
    assert_eq!(pinf.op, "SOFT_MAX");
    oracle::assert_close(&tr.probs, &probs, 1e-4, "route -> ffn_moe_probs-1");

    let (weights, winf) = o.load("ffn_moe_weights-1", 0);
    assert_eq!(winf.op, "GET_ROWS");
    // Raw softmax probabilities, no renormalization — that is what the reference holds.
    oracle::assert_close(&tr.weights, &weights, 1e-4, "route -> ffn_moe_weights-1");

    // Routing reads the router; it must not have touched any expert stack.
    assert!(
        moe::last_touched_experts().is_empty(),
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
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_moe_forward_matches_ggml() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let x = moe_input(&o);
    moe::set_trace_enabled(true);

    let out = plan::moe_ffn(&g, 1, &x).unwrap();
    let tr = moe::last_trace().expect("moe_ffn must leave a trace");
    assert_eq!((out.ne0, out.ne1), (2048, 6), "ffn_out-1 shape");

    let (par, parinf) = o.load("ffn_moe_gate_par-1", 0);
    assert_eq!(parinf.op, "MOE_FUSED_UP_GATE");
    oracle::assert_close(&tr.gate_par, &par, 1e-4, "moe_ffn -> ffn_moe_gate_par-1");

    let (down, dinf) = o.load("ffn_moe_down-1", 0);
    assert_eq!(dinf.op, "MUL_MAT_ID");
    // This one gate is opened above the spec's 1e-4, with the derivation and
    // the FAIL-first evidence the relaxation rule asks for.
    //
    // The reference's fused up/gate kernel accumulates in f32, so its own
    // `gate_par` — the very tensor `MUL_MAT_ID` then quantizes — sits a few
    // 1e-5 from this engine's, and the down quantizer is a discontinuous
    // function of that input (bf16 block scale + integer codes): a couple of
    // input blocks per batch sit close enough to a code or scale boundary
    // that the reference's own noise flips them, and one flipped code
    // perturbs a whole 2048-entry output column. The per-token error
    // signature (a few columns at ~2.8e-4, the rest exact) is that discrete
    // flip pattern, not a smooth scale or accumulation error; the same floor
    // appears whichever activation quantizer feeds the down matmul, so it is
    // a property of the reference, not of this engine's quantizer choice.
    //
    // PIN(2026-09-19): opened 1e-4 -> 4e-4 — the reference's own gate_par
    // noise flips ~1-2 down-quantizer codes per batch; 4e-4 is 1.4x the
    // observed maximum and still fails an f16-scale-in-place-of-bf16 bug at
    // 70x the gate.
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
/// §4.4); this gate keeps that failure mode out of this engine —
/// `moe::last_touched_experts()` is set where the expert bytes are fetched, so a
/// missing empty-bucket skip trips it at "64 distinct experts dequantized".
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_moe_touches_only_routed_experts() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let x = moe_input(&o);
    moe::set_trace_enabled(true);

    plan::moe_ffn(&g, 1, &x).unwrap();
    let tr = moe::last_trace().expect("moe_ffn must leave a trace");
    let touched = moe::last_touched_experts();

    let mut routed: Vec<usize> = tr.ids.iter().map(|&e| e as usize).collect();
    routed.sort_unstable();
    routed.dedup();
    let n_distinct = routed.len();

    assert_eq!(
        touched.len(),
        n_distinct,
        "distinct experts dequantized must equal distinct experts routed to ({}), got {}",
        n_distinct,
        touched.len()
    );
    assert_eq!(
        touched, routed,
        "dequantized an expert no token routed to — the mistral.rs x86_64 failure mode (prior-art §4.4)"
    );
    assert!(
        n_distinct < tr.n_expert,
        "the reference batch routes to {n_distinct} of {} experts; if it ever routes to all, \
         this gate can no longer distinguish bucketed dispatch from touch-all-64",
        tr.n_expert
    );
}

/// The host tier's one-file entry is the composition it replaces: the gate/up
/// group, the down group over the SwiGLU combines and the list-order weighted
/// sum from zero, bit for bit — through the caller's scratch on the first
/// call and again on a second, for a list with a negative weight and a
/// repeated expert, and zeros for an empty list.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_experts_into_matches_group_composition() {
    use model::ops::{GroupInput, matmul_q_group, matmul_q_group_swiglu};

    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let derived = model::arch::deepseek2::derived::Derived::new(&g).unwrap();
    let plan = derived.block_plan(1).unwrap().moe().unwrap();
    let x = moe_input(&o);
    let x1 = Tensor2::from_vec(x.ne0, 1, x.col(0).to_vec());
    let list = [(3u32, 0.25f32), (17, -0.5), (60, 1.25), (3, 0.125)];

    let gu_ws: Vec<&gguf::TensorInfo> = list
        .iter()
        .flat_map(|&(e, _)| [&plan.gate_views[e as usize], &plan.up_views[e as usize]])
        .collect();
    let gu = matmul_q_group(&g, &gu_ws, &vec![&x1; gu_ws.len()]).unwrap();
    let down_ws: Vec<&gguf::TensorInfo> = list
        .iter()
        .map(|&(e, _)| &plan.down_views[e as usize])
        .collect();
    let srcs: Vec<GroupInput> = (0..list.len())
        .map(|i| GroupInput::Swiglu(&gu[2 * i], &gu[2 * i + 1]))
        .collect();
    let (downs, _) = matmul_q_group_swiglu(&g, &down_ws, &srcs).unwrap();
    let mut want = vec![0.0f32; x1.ne0];
    for (&(_, w), d) in list.iter().zip(&downs) {
        for (o, &dv) in want.iter_mut().zip(d.col(0)) {
            *o += w * dv;
        }
    }

    let mut scratch = moe::HostScratch::new(x1.ne0, plan.meta.ff);
    for call in 0..2 {
        let mut got = vec![f32::NAN; x1.ne0];
        moe::experts_into(&g, plan, &x1, &list, &mut got, &mut scratch).unwrap();
        let diffs = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "call {call}: experts_into must equal the group composition bit for bit"
        );
    }
    let mut empty = vec![f32::NAN; x1.ne0];
    moe::experts_into(&g, plan, &x1, &[], &mut empty, &mut scratch).unwrap();
    assert!(
        empty.iter().all(|v| v.to_bits() == 0),
        "an empty list writes zeros"
    );
    eprintln!(
        "experts_into == group composition, bit for bit: {} experts, two calls; empty list zeros",
        list.len()
    );
}
