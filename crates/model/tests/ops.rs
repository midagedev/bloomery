//! Lead-owned gate for the two primitives every other module stands on. If these are wrong,
//! four rounds fail for a reason that is not theirs.
//!
//! `hw_` prefix: needs the box (the model file and the oracle set), excluded by default.
#[path = "common/oracle.rs"]
mod oracle;

use gguf::GgmlType;
use model::ops::{Tensor2, f32_tensor, matmul_q, rms_norm};

#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_rms_norm_matches_ggml() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

    let (inp, inf) = o.load("inp_embd", 0);
    let x = Tensor2::from_vec(inf.ne[0] as usize, inf.ne[1] as usize, inp);

    let gain_t = g.find("blk.0.attn_norm.weight").unwrap();
    let gain = f32_tensor(&g, gain_t).unwrap();

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
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

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
    // 1e-4 on values that reach 18: the residual is f32 accumulation order
    // against ggml's integer sum.
    oracle::assert_close(&got.data, &want, 1e-4, "matmul_q -> q-0");
}

/// The dispatch proof for the qdot wiring: `matmul_q` must route Q3_K with k
/// a multiple of 256 through the fused kernel. The two paths are known NOT to
/// agree bit for bit — the fused kernel is the more accurate one — so this
/// test proves the move HAPPENED, not that it did not: bit equality against
/// the fused composition (ground A), at least one differing element against
/// the old scalar composition (ground B). Q5_1 gets the same two grounds
/// against blk.0.ffn_down (k = 10944 = 342 x 32, not a multiple of 256 — the
/// shape the per-type `k_granularity` contract exists for); with every quant
/// type in the model fused, the failure this watches for is the fused path
/// silently not firing.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_matmul_q_q3k_fused_dispatch() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

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

    // Q5_1 dispatch proof: blk.0.ffn_down, the same two grounds. The input is
    // synthetic (all ones, k-matched), so this needs no oracle entry.
    let wf = g.find("blk.0.ffn_down.weight").unwrap();
    assert_eq!(wf.ty, GgmlType::Q5_1, "this is the Q5_1 dispatch proof");
    let kf = wf.dims[0] as usize;
    let nf = wf.dims[1] as usize;
    let xf = Tensor2::from_vec(kf, 1, vec![1.0f32; kf]);
    let gotf = matmul_q(&g, wf, &xf).unwrap();
    let fbytes = g.data(wf).unwrap();
    let frow_bytes = fbytes.len() / nf;
    // Ground A' — bit-identical to the fused composition (tail blocks
    // included: k % 128 = 64 leaves two of them).
    let fcb = qdot::col_bytes(GgmlType::Q5_1, kf);
    let mut facol = vec![0u8; fcb];
    qdot::quantize_col(GgmlType::Q5_1, xf.col(0), &mut facol);
    for r in 0..nf {
        let src = &fbytes[r * frow_bytes..(r + 1) * frow_bytes];
        let v = qdot::dot_row(GgmlType::Q5_1, src, &facol, kf).unwrap();
        assert_eq!(
            gotf.data[r].to_bits(),
            v.to_bits(),
            "row {r}: must be bit-identical to the fused composition"
        );
    }
    eprintln!("ground A': {nf} rows bit-identical to the qdot Q5_1 composition");
    // Ground B' — must differ somewhere from the OLD scalar composition, or
    // the fused path never ran.
    let mut fq = vec![0.0f32; kf];
    gguf::quantize_activations(GgmlType::Q5_1, xf.col(0), &mut fq);
    let mut frow = vec![0.0f32; kf];
    let mut fdiffs = 0usize;
    for r in 0..nf {
        let src = &fbytes[r * frow_bytes..(r + 1) * frow_bytes];
        gguf::dequant_row(GgmlType::Q5_1, src, &mut frow).unwrap();
        let mut acc = 0.0f32;
        for i in 0..kf {
            acc += frow[i] * fq[i];
        }
        if acc.to_bits() != gotf.data[r].to_bits() {
            fdiffs += 1;
        }
    }
    assert!(
        fdiffs >= 1,
        "bit-identical to the scalar composition — the Q5_1 fused path did not \
         run, so ground A' proved nothing ({nf} rows compared)"
    );
    eprintln!(
        "ground B': {fdiffs} of {nf} rows differ from the scalar composition — Q5_1 moved onto the fused path"
    );
}

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_matmul_q_batch_matches_sequential() {
    // The dispatch proof: the batched primitive is the per-pair matmul_q to
    // the last bit, on the tensors it actually serves (the routed-expert
    // stacks) with the shapes that actually differ between pairs (bucket
    // sizes — here 2 and 5 columns, mixed on purpose so a pair that read its
    // neighbor's activations fails on length alone).
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

    // Slice the expert stack exactly the way moe.rs's expert_view does; the
    // test rebuilds it because the batch must equal the sequential view, not
    // because the slicing itself is under test.
    let stack = g.find("blk.1.ffn_gate_exps.weight").unwrap();
    assert_eq!(
        stack.ty,
        GgmlType::Q3_K,
        "the fused path must be the one exercised"
    );
    assert_eq!(
        stack.dims.len(),
        3,
        "expert stacks are dims len 3: k, n, n_expert"
    );
    let k = stack.dims[0] as usize;
    let n = stack.dims[1] as usize;
    let n_expert = stack.dims[2] as usize;
    assert!(
        k.is_multiple_of(256),
        "fused dispatch requires k % 256 == 0"
    );
    let per = stack.nbytes / n_expert as u64;
    let view = |e: u64| gguf::TensorInfo {
        name: stack.name.clone(),
        dims: vec![stack.dims[0], stack.dims[1]],
        ty: stack.ty,
        offset: stack.offset + e * per,
        nbytes: per,
    };

    // Activations: reuse the oracle's first-block input so the values are
    // real model activations, then split into per-expert buckets of 2 and 5
    // columns (subsets, like routing produces).
    let o = oracle::Oracle::open();
    let (xs, xinf) = o.load("attn_norm-0", 0);
    let x = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);
    let colset = |idx: &[usize]| {
        let mut xb = Tensor2::zeros(x.ne0, idx.len());
        for (i, &t) in idx.iter().enumerate() {
            xb.col_mut(i).copy_from_slice(x.col(t));
        }
        xb
    };
    let experts = [0usize, 1, 2, 3];
    let cols = [vec![0, 1], vec![2, 3, 4], vec![0, 2, 4, 5, 1], vec![3]];

    let xbs: Vec<Tensor2> = cols.iter().map(|c| colset(c)).collect();
    let views: Vec<gguf::TensorInfo> = experts.iter().map(|&e| view(e as u64)).collect();

    // Reference: one dispatch per pair, exactly what a sequential expert loop
    // runs.
    let seq: Vec<Tensor2> = views
        .iter()
        .zip(&xbs)
        .map(|(w, xb)| matmul_q(&g, w, xb).unwrap())
        .collect();

    // The batch: one dispatch for all four.
    let ws: Vec<&gguf::TensorInfo> = views.iter().collect();
    let xref: Vec<&Tensor2> = xbs.iter().collect();
    let got = model::ops::matmul_q_batch(&g, &ws, &xref).unwrap();

    assert_eq!(got.len(), seq.len(), "one output per pair");
    for (p, (want, got)) in seq.iter().zip(&got).enumerate() {
        assert_eq!([got.ne0, got.ne1], [want.ne0, want.ne1], "pair {p} shape");
        let diffs = want
            .data
            .iter()
            .zip(&got.data)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diffs, 0,
            "pair {p}: batched must be bit-identical to sequential"
        );
    }
    eprintln!(
        "batch of {} pairs x {n} rows (2/3/5/1 columns) bit-identical to sequential",
        experts.len()
    );
}

/// The heterogeneous dispatch proof: `matmul_q_group` over mixed types,
/// shapes and inputs is the per-pair `matmul_q` to the last bit — the same
/// kernels on the same bytes, only the dispatch shape changed. The set is
/// the step's own mixture over this file's real quant blend: the attention
/// pair (Q3_K, `n` 576 and 3072), the F32 router, blk.6's Q4_K
/// `attn_output`, the blk.1 shexp gate and a Q5_0 routed-down expert view on
/// its own 1408-wide input — six different `n`, three encodings. Also pins
/// the activation-sharing rule's observable: one quantization slot per
/// DISTINCT (input, encoding) — same-format pairs collapse (across weight
/// types), different encodings do not.
#[test]
#[ignore = "hw: needs the box, the model file and $BLOOMERY_DATA/ref"]
fn hw_matmul_q_group_matches_sequential() {
    let o = oracle::Oracle::open();
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();

    let (xs, xinf) = o.load("attn_norm-0", 0);
    let x6 = Tensor2::from_vec(xinf.ne[0] as usize, xinf.ne[1] as usize, xs);
    let colset = |idx: &[usize]| {
        let mut xb = Tensor2::zeros(x6.ne0, idx.len());
        for (i, &t) in idx.iter().enumerate() {
            xb.col_mut(i).copy_from_slice(x6.col(t));
        }
        xb
    };
    let x5 = colset(&[0, 1, 2, 3, 4]);
    let x1 = colset(&[0]);

    let wq = g.find("blk.0.attn_q.weight").unwrap();
    let wa = g.find("blk.0.attn_kv_a_mqa.weight").unwrap();
    let router = g.find("blk.1.ffn_gate_inp.weight").unwrap();
    let shg = g.find("blk.1.ffn_gate_shexp.weight").unwrap();
    let wo6 = g.find("blk.6.attn_output.weight").unwrap();
    let outw = g.find("output.weight").unwrap();
    assert_eq!(wq.ty, GgmlType::Q3_K, "the mixed set needs the real types");
    assert_eq!(wa.ty, GgmlType::Q3_K);
    assert_eq!(router.ty, GgmlType::F32);
    assert_eq!(shg.ty, GgmlType::Q3_K);
    assert_eq!(wo6.ty, GgmlType::Q4_K);
    assert_eq!(outw.ty, GgmlType::Q6_K);
    // The routed-down view sliced exactly the way moe.rs's expert_view does;
    // its k is the expert width, so it reads its own input below.
    let stack = g.find("blk.6.ffn_down_exps.weight").unwrap();
    assert_eq!(stack.ty, GgmlType::Q5_0);
    let per = stack.nbytes / stack.dims[2];
    let e0 = gguf::TensorInfo {
        name: stack.name.clone(),
        dims: vec![stack.dims[0], stack.dims[1]],
        ty: stack.ty,
        offset: stack.offset + per,
        nbytes: per,
    };
    let ns: Vec<u64> = [wq, wa, router, shg, wo6, &e0]
        .iter()
        .map(|w| w.dims[1])
        .collect();
    let distinct = ns.iter().collect::<std::collections::BTreeSet<_>>();
    assert!(distinct.len() >= 5, "the set must span different n: {ns:?}");

    // The 1408-wide input for the routed-down pair — synthetic like the
    // fused-dispatch test's, non-zero so the dots mean something.
    let yset = |m: usize| {
        Tensor2::from_vec(
            e0.dims[0] as usize,
            m,
            (0..e0.dims[0] as usize * m)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.031)
                .collect(),
        )
    };
    let y5 = yset(5);
    let y1 = yset(1);

    let ws: Vec<&gguf::TensorInfo> = vec![wq, wa, router, shg, wo6, &e0];
    let check = |label: &str, x: &Tensor2, y: &Tensor2| {
        let xs_ref: Vec<&Tensor2> = vec![x, x, x, x, x, y];
        let got = model::ops::matmul_q_group(&g, &ws, &xs_ref).unwrap();
        assert_eq!(got.len(), ws.len(), "one output per pair ({label})");
        for (p, (w, xin)) in ws.iter().zip(&xs_ref).enumerate() {
            let want = matmul_q(&g, w, xin).unwrap();
            assert_eq!(
                [got[p].ne0, got[p].ne1],
                [want.ne0, want.ne1],
                "pair {p} ({w:?}) shape ({label})"
            );
            let diffs = want
                .data
                .iter()
                .zip(&got[p].data)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                diffs, 0,
                "pair {p} ({w:?}): group must be bit-identical to sequential ({label})"
            );
        }
    };
    check("ne1=5", &x5, &y5);
    check("ne1=1", &x1, &y1);
    eprintln!(
        "group of {} mixed pairs (Q3_K x3, Q4_K, Q5_0, F32; n spans {:?}) \
         bit-identical to sequential at ne1=5 and ne1=1",
        ws.len(),
        distinct.iter().collect::<Vec<_>>(),
    );

    // The sharing rule's observable. The mixed group lands in FOUR slots:
    // Q8K over x (wq, wa, shg), the F32 round trip over x (router), Q8_2X4
    // over x (wo6), Q8_2X4 over y (e0 — same format as wo6, different input
    // and stride).
    assert_eq!(
        model::ops::last_quant_slots(),
        4,
        "one slot per (input, encoding) in the mixed group"
    );
    // A same-encoding pair on one input collapses — even across weight types
    // and different n: Q4_K and Q6_K both quantize to Q8_2X4 at k = 2048.
    let got = model::ops::matmul_q_group(&g, &[wo6, outw], &[&x1, &x1]).unwrap();
    assert_eq!(model::ops::last_quant_slots(), 1, "cross-type share");
    let w_4k = matmul_q(&g, wo6, &x1).unwrap();
    let w_6k = matmul_q(&g, outw, &x1).unwrap();
    for (p, want) in [w_4k, w_6k].iter().enumerate() {
        let diffs = want
            .data
            .iter()
            .zip(&got[p].data)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(diffs, 0, "shared-slot pair {p} must stay bit-identical");
    }
    // Different formats on one input stay apart: Q3_K (Q8K) vs Q4_K.
    model::ops::matmul_q_group(&g, &[wq, wo6], &[&x1, &x1]).unwrap();
    assert_eq!(
        model::ops::last_quant_slots(),
        2,
        "different encodings never share"
    );

    // A single-pair group is the single-pair call.
    let one = model::ops::matmul_q_group(&g, &[wa], &[&x1]).unwrap();
    let want_one = matmul_q(&g, wa, &x1).unwrap();
    assert_eq!([one[0].ne0, one[0].ne1], [want_one.ne0, want_one.ne1]);
    let diffs = want_one
        .data
        .iter()
        .zip(&one[0].data)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(diffs, 0, "single-pair group must equal matmul_q");

    // The empty group is Ok(empty), the empty batch's contract.
    assert!(
        model::ops::matmul_q_group(&g, &[], &[]).unwrap().is_empty(),
        "the empty group is Ok(empty)"
    );

    // A mismatched pair errors exactly as the single-pair call does.
    let bad = Tensor2::zeros(576, 1);
    let e_group = model::ops::matmul_q_group(&g, &[wq], &[&bad]).unwrap_err();
    let e_one = matmul_q(&g, wq, &bad).unwrap_err();
    match (e_group, e_one) {
        (
            model::ModelError::Shape {
                what: w1,
                want_ne0: a1,
                want_ne1: b1,
                got_ne0: c1,
                got_ne1: d1,
            },
            model::ModelError::Shape {
                what: w2,
                want_ne0: a2,
                want_ne1: b2,
                got_ne0: c2,
                got_ne1: d2,
            },
        ) => {
            assert_eq!(
                (w1, a1, b1, c1, d1),
                (w2, a2, b2, c2, d2),
                "the group's per-pair error is matmul_q's own"
            );
        }
        _ => panic!("both calls must return the same Shape error"),
    }
    eprintln!("sharing, single-pair, empty and error-parity contracts hold");
}
