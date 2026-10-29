//! The clause bodies of a decode owner no fixture carries yet: every body
//! the owner instantiates, run on seeded synthetic rows at the family's real
//! shapes ([`synthetic::matrix`]: random codes under small finite scales)
//! and held to the same f64 oracle and bands as the fixture clauses. The
//! quant is a parameter; `clauses_q8_0` and `clauses_q5k` name the clauses.
//!
//! The shapes are the Qwen3.6 layer's: the mat-vec's K of 2048 (the qkv, q
//! and gate projections), 4096 (the gated delta net's out projection) and 512
//! (the shared expert's down), the experts' ff of 512, the head's 248,320 rows
//! of 2048 and the embedding's table of the same shape.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::kernels::{GemmMode, PARTIALS_PER_HEAD, Quant};

use super::clauses_experts::{
    FOLD_ROWS, FOLD_TAIL_ROWS, check_fold_weights, check_pair_weights, fold_bad_id, pair_faults,
    synthetic_fold_for, synthetic_norm, synthetic_pair_for,
};
use super::clauses_gemm::{Case, SEG_COLS, SegGroup, dense_arms, seg_groups};
use super::clauses_norm::head_partials;
use super::common::{self, LAYER};
use super::fixture::OUTPUT_ROWS;
use super::oracle::{self, Kind};
use super::synthetic;
use super::{
    ClauseRun, Harness, HeadNormLaunch, Launch, ModeSpec, SegLaunch, clauses_embed, clauses_head,
    clauses_route,
};

/// The epsilon every synthetic norm takes.
const EPS: f32 = 1e-6;

/// The mat-vec shapes the store and residual cases run: a name, the columns
/// and the rows.
const SHAPES: [(&str, usize, usize); 4] = [
    ("attn_qkv", 2048, 8192),
    ("ssm_out", 4096, 2048),
    ("ffn_down_shexp", 512, 2048),
    ("attn_q", 2048, 8192),
];

/// The head's last rows: a count no threadgroup divides.
const HEAD_TAIL: usize = 1153;

/// A quant's seed base: two quants' rows differ.
fn salt(quant: Quant) -> u64 {
    (quant.block_bytes() as u64) << 24
}

/// A positive synthetic gain: a dropped gain shows.
pub(super) fn gain(cols: usize, seed: u64) -> Vec<f32> {
    oracle::activations(cols, seed, 0.5)
        .iter()
        .map(|g| 1.0 + g.abs())
        .collect()
}

/// The store dot at the layer's shapes and the head's tail rows.
pub fn matvec(h: &Harness<'_>, quant: Quant) -> Clause {
    let mut shapes: Vec<(String, usize, usize)> = SHAPES
        .iter()
        .map(|&(name, cols, rows)| (name.to_string(), cols, rows))
        .collect();
    shapes.push((
        format!("output tail {HEAD_TAIL}"),
        h.geom().hidden,
        HEAD_TAIL,
    ));
    let mut parts = Vec::new();
    for (i, (name, cols, rows)) in shapes.into_iter().enumerate() {
        let weights = synthetic::matrix(quant, rows, cols, salt(quant) + i as u64);
        let x = oracle::activations(cols, 0x5A34 + i as u64, 1.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &weights,
                x: &x,
                y0: &common::sentinel_rows(rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        parts.push(format!("{name} {rows}x{cols} {worst:.3}"));
    }
    common::noted(parts)
}

/// Row counts no threadgroup divides: every row written, no word past the
/// last.
pub fn row_tail(h: &Harness<'_>, quant: Quant) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    let cols = 2048;
    let all = synthetic::matrix(quant, 4095, cols, salt(quant) + 0x10);
    let row_bytes = cols / quant.values_per_block() * quant.block_bytes();
    let x = oracle::activations(cols, 0x5A60, 1.0);
    for rows in counts {
        common::run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &all[..rows * row_bytes],
                x: &x,
                y0: &common::sentinel_rows(rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{rows} rows of {cols}: {e}"))?;
    }
    Ok(Verdict::Noted(format!(
        "{} row counts, every row written, no word past the last",
        counts.len()
    )))
}

/// The norm on load at the layer's shapes and the head's, a zero input with
/// a positive epsilon, the fault inputs (a NaN input, an infinite one, a zero
/// input with a zero epsilon: the norm site and NaN rows), and the residual
/// epilogue.
pub fn norm(h: &Harness<'_>, quant: Quant) -> Clause {
    let mut cases: Vec<(String, usize, usize)> = SHAPES[..3]
        .iter()
        .map(|&(name, cols, rows)| (name.to_string(), cols, rows))
        .collect();
    cases.push(("output".to_string(), h.geom().hidden, 4096));
    let mut parts = Vec::new();
    for (i, (name, cols, rows)) in cases.iter().enumerate() {
        let weights = synthetic::matrix(quant, *rows, *cols, salt(quant) + 0x20 + i as u64);
        let gain = gain(*cols, 0x5AA1 + i as u64);
        let x = oracle::activations(*cols, 0x5A40 + i as u64, 3.7);
        let worst = common::run_checked(
            h,
            &Launch {
                quant,
                rows: *rows,
                cols: *cols,
                weights: &weights,
                x: &x,
                y0: &common::sentinel_rows(*rows),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps: EPS,
                    layer: LAYER,
                },
            },
        )
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        parts.push(format!("{name} {rows}x{cols} {worst:.3}"));
    }

    let (cols, rows) = (4096, 2048);
    let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x21);
    let g = gain(cols, 0x5AB0);
    // A zero input norms to zero with a positive epsilon: not a fault.
    common::run_checked(
        h,
        &Launch {
            quant,
            rows,
            cols,
            weights: &weights,
            x: &vec![0f32; cols],
            y0: &common::sentinel_rows(rows),
            mode: ModeSpec::Norm {
                gain: &g,
                eps: EPS,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("a zero input with eps {EPS}: {e}"))?;
    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(cols, 1, 1.0), EPS),
        ("an infinite input", oracle::activations(cols, 2, 1.0), EPS),
        ("a zero input and a zero epsilon", vec![0f32; cols], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[cols - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.matvec(&Launch {
            quant,
            rows,
            cols,
            weights: &weights,
            x: &x,
            y0: &common::sentinel_rows(rows),
            mode: ModeSpec::Norm {
                gain: &g,
                eps,
                layer: LAYER,
            },
        })?;
        common::expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;

    // The residual epilogue: the store's rows plus the output's own.
    for (i, &(name, cols, rows)) in SHAPES[..3].iter().enumerate() {
        let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x28 + i as u64);
        let x = oracle::activations(cols, 0x5A50 + i as u64, 1.0);
        let resid = oracle::activations(rows, 0x5BE5 + i as u64, 2.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &weights,
                x: &x,
                y0: &resid,
                mode: ModeSpec::Residual,
            },
        )
        .map_err(|e| format!("resid {name} {rows}x{cols}: {e}"))?;
        parts.push(format!("resid {name} {rows}x{cols} {worst:.3}"));
    }
    common::noted(parts)
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone (the norm on load), in the layer's projection groups.
pub fn seg(h: &Harness<'_>, quant: Quant) -> Clause {
    let cols = 2048;
    let groups: [&[(&str, usize)]; 2] = [
        &[("attn_q", 8192), ("attn_k", 512), ("attn_v", 512)],
        &[("attn_qkv", 8192), ("attn_gate", 4096)],
    ];
    let g = gain(cols, 0x5E80);
    let mut parts = Vec::new();
    for (i, tensors) in groups.iter().enumerate() {
        let x = oracle::activations(cols, 0x5E86 + i as u64, 1.0);
        let owned: Vec<Vec<u8>> = tensors
            .iter()
            .enumerate()
            .map(|(j, (_, n))| {
                synthetic::matrix(quant, *n, cols, salt(quant) + 0x30 + (8 * i + j) as u64)
            })
            .collect();
        let segments: Vec<(usize, &[u8])> = tensors
            .iter()
            .zip(&owned)
            .map(|((_, n), w)| (*n, w.as_slice()))
            .collect();
        let rows: usize = tensors.iter().map(|(_, n)| n).sum();
        let got = h.seg(&SegLaunch {
            quant,
            cols,
            segments: segments.clone(),
            x: &x,
            y0: &common::sentinel_rows(rows),
            gain: &g,
            eps: EPS,
            layer: LAYER,
        })?;
        if let Some(fault) = h.fault()? {
            return Err(format!("the segmented launch raised {fault}"));
        }
        let mut at = 0usize;
        for (j, (n, bytes)) in segments.iter().enumerate() {
            let alone = h.matvec(&Launch {
                quant,
                rows: *n,
                cols,
                weights: bytes,
                x: &x,
                y0: &common::sentinel_rows(*n),
                mode: ModeSpec::Norm {
                    gain: &g,
                    eps: EPS,
                    layer: LAYER,
                },
            })?;
            if let Some((k, a, b)) = common::first_diff(&got[at..at + n], &alone) {
                return Err(format!(
                    "{} segment {j} row {k}: the segment read {a:?}, the tensor alone read {b:?}",
                    tensors[j].0
                ));
            }
            at += n;
        }
        parts.push(format!("{} {} segments", tensors[0].0, tensors.len()));
    }
    common::noted(parts)
}

/// The head-normed residual mat-vec: the gated numerator times the scale of
/// the head each value sits in, against the f64 oracle, and a partial and a
/// numerator that are not finite (the gate site, every row NaN).
pub fn headnorm(h: &Harness<'_>, quant: Quant) -> Clause {
    let delta = &h.geom().delta;
    let cols = delta.head_cols * delta.value_heads;
    let rows = 2048;
    let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x40);
    let g = oracle::activations(cols, 0x5E88, 1.0);
    let partials = head_partials(delta);
    let x = common::head_scaled(
        &g,
        &partials,
        delta.value_heads,
        delta.head_cols,
        PARTIALS_PER_HEAD,
        EPS,
    );
    let reference = oracle::reference(quant, &weights, rows, cols, &x)?;
    let resid = oracle::activations(rows, 0x5E89, 2.0);
    let launch = |g: &[f32], partials: &[f32]| {
        h.headnorm(&HeadNormLaunch {
            quant,
            rows,
            cols,
            weights: &weights,
            g,
            partials,
            y0: &resid,
            eps: EPS,
            layer: LAYER,
        })
    };
    h.clear_fault()?;
    let got = launch(&g, &partials)?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the launch raised {fault}"));
    }
    let kind = Kind::HeadNorm(oracle::HeadGeometry {
        heads: delta.value_heads,
        cols_per_head: delta.head_cols,
        partials_per_head: PARTIALS_PER_HEAD,
    });
    let band = oracle::band(quant, cols / quant.values_per_block(), kind);
    let worst = oracle::check(&got, &reference, band, kind, Some(&resid))?;

    let mut bad_partials = partials.clone();
    bad_partials[5 * PARTIALS_PER_HEAD + 3] = f32::INFINITY;
    let mut bad_g = g.clone();
    bad_g[17] = f32::NAN;
    for (what, g, partials) in [
        ("a partial that is not finite", &g, &bad_partials),
        ("a gated numerator that is not finite", &bad_g, &partials),
    ] {
        h.clear_fault()?;
        let got = launch(g, partials)?;
        common::expect_fault(h, what, FaultSite::LinearGate)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst:.3}; a partial and a gated numerator that are not \
         finite each raised the gate site, every row NaN"
    )))
}

/// The gate-up pair over four experts' stacks and the shared pair: the
/// oracle band, an id past the stack, and the norm's fault inputs.
pub fn pair(h: &Harness<'_>, quant: Quant) -> Clause {
    let weights = synthetic_pair_for(&h.geom().experts, quant, salt(quant) + 0x50);
    let norm = synthetic_norm(EPS);
    let worst = check_pair_weights(h, quant, &weights, &norm)?;
    pair_faults(h, quant, &weights, &norm)?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; ids -1 and the stack length raised expert_id, \
         NaN in that slot only; the norm's fault inputs raised norm_quant, NaN in all nine \
         slots' rows"
    )))
}

/// The folded down over four experts' stacks and the shared down: the oracle
/// band at the engine's rows and at a row tail, and a bad id's NaN rows.
pub fn fold(h: &Harness<'_>, quant: Quant) -> Clause {
    let experts = &h.geom().experts;
    let whole = synthetic_fold_for(experts, quant, FOLD_ROWS, salt(quant) + 0x58);
    let worst = check_fold_weights(h, quant, &whole, FOLD_ROWS)?;
    let tail = synthetic_fold_for(experts, quant, FOLD_TAIL_ROWS, salt(quant) + 0x5A);
    let tail = check_fold_weights(h, quant, &tail, FOLD_TAIL_ROWS)?;
    fold_bad_id(h, &whole, quant)?;
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; tail {FOLD_TAIL_ROWS} rows: {tail}; a bad id \
         writes NaN and raises expert_id"
    )))
}

/// The head over the whole vocabulary: every logit against the norm-on-load
/// oracle band, the step id the host argmax of the read-back logits.
pub fn head(h: &Harness<'_>, quant: Quant) -> Clause {
    let hidden = h.geom().hidden;
    let weights = synthetic::matrix(quant, OUTPUT_ROWS, hidden, salt(quant) + 0x60);
    let x = oracle::activations(hidden, 0x5EAD, 0.5);
    let gain = gain(hidden, 0x5EAE);
    h.clear_fault()?;
    let out = clauses_head::run(
        h,
        clauses_head::HeadRun {
            quant,
            weights: &weights,
            x: &x,
            gain: &gain,
            rows: OUTPUT_ROWS,
            k: hidden,
            eps: EPS,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("the head raised {f}"));
    }
    let xhat = oracle::rms_norm(&x, &gain, EPS);
    let reference = oracle::reference(quant, &weights, OUTPUT_ROWS, hidden, &xhat)?;
    let band = oracle::band(quant, hidden / quant.values_per_block(), Kind::Norm);
    let worst = oracle::check(&out.logits, &reference, band, Kind::Norm, None)?;
    let (_, want) = clauses_head::host_argmax(&out.logits)?;
    if out.id != want {
        return Err(format!(
            "the step id is {}, the logits select {want}",
            out.id
        ));
    }
    Ok(Verdict::Noted(format!(
        "{OUTPUT_ROWS}x{hidden} worst {worst:.3}; step id {want}, the logits' argmax"
    )))
}

/// The embedding over a whole-vocabulary table: the owner's value decode
/// bit-equal to the host reader, and the ids past the vocabulary.
pub fn embed(h: &Harness<'_>, quant: Quant) -> Clause {
    let table = synthetic::matrix(quant, OUTPUT_ROWS, h.geom().hidden, salt(quant) + 0x68);
    Ok(Verdict::Noted(clauses_embed::embed_table(
        h,
        quant,
        &table,
        OUTPUT_ROWS,
    )?))
}

/// The dense GEMM's synthetic tensors: the layer's shapes at the activation
/// row counts that exercise the tile's edges, a strided case, and the
/// head's tail.
struct Dense {
    qkv: Vec<u8>,
    ssm_out: Vec<u8>,
    down: Vec<u8>,
    tail: Vec<u8>,
}

fn dense(quant: Quant) -> Dense {
    let seed = salt(quant) + 0x70;
    Dense {
        qkv: synthetic::matrix(quant, 8192, 2048, seed),
        ssm_out: synthetic::matrix(quant, 2048, 4096, seed + 1),
        down: synthetic::matrix(quant, 2048, 512, seed + 2),
        tail: synthetic::matrix(quant, HEAD_TAIL, 2048, seed + 3),
    }
}

fn dense_cases(quant: Quant, d: &Dense) -> Vec<Case<'_>> {
    let case = |label: &str, weights, rows, cols, acts, x_stride, y_stride, seed| Case {
        label: label.to_string(),
        weights,
        quant,
        rows,
        cols,
        acts,
        x_stride,
        y_stride,
        seed,
    };
    vec![
        case(
            "attn_qkv 8192x2048 r9",
            &d.qkv,
            8192,
            2048,
            9,
            2048,
            8192,
            0x6E81,
        ),
        case(
            "ssm_out 2048x4096 r33 strided",
            &d.ssm_out,
            2048,
            4096,
            33,
            4096 + 16,
            2048 + 4,
            0x6E82,
        ),
        case(
            "ffn_down_shexp 2048x512 r512",
            &d.down,
            2048,
            512,
            512,
            512,
            2048,
            0x6E83,
        ),
        case(
            "output tail 1153x2048 r33",
            &d.tail,
            HEAD_TAIL,
            2048,
            33,
            2048,
            1156,
            0x6E84,
        ),
    ]
}

/// The dense GEMM on `h`'s kernels: both epilogues in both operands, every
/// element within the band of `h`'s GEMM path.
pub fn gemm(h: &Harness<'_>, quant: Quant) -> Clause {
    let d = dense(quant);
    dense_arms(h, &dense_cases(quant, &d))
}

/// The segmented GEMM on `h`'s kernels: every segment the same bits as its
/// tensor alone, a tail segment included, both epilogues in both operands.
pub fn gemm_seg(h: &Harness<'_>, quant: Quant) -> Clause {
    let seed = salt(quant) + 0x78;
    let qkv = synthetic::matrix(quant, 8192, SEG_COLS, seed);
    let gate = synthetic::matrix(quant, 4096, SEG_COLS, seed + 1);
    let tail = synthetic::matrix(quant, 511, SEG_COLS, seed + 2);
    let segments = vec![(8192, qkv.as_slice()), (4096, &gate[..]), (511, &tail[..])];
    let groups = [GemmMode::Store, GemmMode::Residual].map(|mode| SegGroup {
        quant,
        segments: segments.clone(),
        mode,
    });
    Ok(Verdict::Noted(format!(
        "{} kernels, every segment the same bits as its tensor alone: {}",
        h.kernels.gemm_path().name(),
        seg_groups(h, &groups)?.join(", ")
    )))
}

/// The grouped GEMM on `h`'s kernels over every expert's own synthetic rows:
/// the pair and the down in both operands within the band, runs ending
/// inside a SIMD group's blocks, the half range's fault, and a tile past the
/// tile's slots.
pub fn grouped(h: &Harness<'_>, quant: Quant) -> Clause {
    clauses_route::grouped_synthetic(h, quant, salt(quant) + 0x80)
}

/// The clauses on the gate's own kernels, in the token's order.
const BODIES: [fn(&Harness<'_>, Quant) -> Clause; 12] = [
    matvec, row_tail, norm, seg, headnorm, pair, fold, head, embed, gemm, gemm_seg, grouped,
];

/// The clauses on the tensor-op kernels.
const TENSOR_BODIES: [fn(&Harness<'_>, Quant) -> Clause; 3] = [gemm, gemm_seg, grouped];

/// `quant`'s clauses under `names`, [`BODIES`]' order. Each starts from a
/// clear fault word and leaves it clear, whatever its body raised: the
/// clauses after these do not clear it.
pub fn table<'a>(
    h: &'a Harness<'a>,
    quant: Quant,
    names: [&'static str; 12],
) -> Vec<(&'static str, ClauseRun<'a>)> {
    names
        .into_iter()
        .zip(BODIES)
        .map(|(name, body)| {
            let run = move || {
                h.clear_fault()?;
                let verdict = body(h, quant);
                h.clear_fault()?;
                verdict
            };
            (name, Box::new(run) as ClauseRun<'a>)
        })
        .collect()
}

/// `quant`'s tensor-op clauses under `names`, [`TENSOR_BODIES`]' order, on
/// `tensor` (or why this device has none). Each starts from a clear fault
/// word.
pub fn tensor_table<'a>(
    tensor: &'a Result<Harness<'a>, String>,
    quant: Quant,
    names: [&'static str; 3],
) -> Vec<(&'static str, ClauseRun<'a>)> {
    names
        .into_iter()
        .zip(TENSOR_BODIES)
        .map(|(name, body)| {
            let run = move || match tensor {
                Err(e) => Err(format!("the tensor-op GEMMs do not build here: {e}")),
                Ok(h) => {
                    h.clear_fault()?;
                    body(h, quant)
                }
            };
            (name, Box::new(run) as ClauseRun<'a>)
        })
        .collect()
}
