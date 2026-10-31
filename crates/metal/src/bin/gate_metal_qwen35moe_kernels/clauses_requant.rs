//! The requantized affine clauses: MLX's affine decode owners the split holds
//! no typed site of — A8 (`quant.metal`'s `A8`, one-byte codes under BF16
//! scales and biases), A4H and A8H (the A4 and A8 rules over F16 scales and
//! biases) — against the f64 oracle, over the same fixture split the A4
//! clauses read with its tensors requantized as the type under test
//! (`fixture.rs`'s requantizer — the split holds its typed sites at MLX_A4).
//! One body for every type, the type a parameter: A4's own shapes and fault
//! inputs, every entry the owner instantiates: the five mat-vec modes, the
//! pair, the fold, the head, the embedding, the dense and segmented GEMMs
//! (both operands, the half-row entries among them) and the grouped GEMM. The
//! bands are A4's own derivation (`oracle.rs`: the same store-path count and
//! norm chain, the same two-term magnitude), because each owner's evaluation
//! tree is A4's, over one-byte codes for the 8-bit types and over a pair that
//! widens exactly for the H types.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::kernels::{GemmMode, PARTIALS_PER_HEAD, Quant};

use super::clauses_a4::with;
use super::clauses_embed;
use super::clauses_experts::{self, check_fold_quant, check_pair_quant, ffn_norm, fold_bad_id};
use super::clauses_gemm::{self, Case, SEG_COLS, SegGroup, dense_arms, run_cases, seg_groups};
use super::clauses_head;
use super::clauses_norm::head_partials;
use super::common::{self, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::{ClauseRun, Harness, HeadNormLaunch, Launch, ModeSpec, SegLaunch};

/// The tensors the mat-vec clauses read: A4's Qwen3.6 row, `(name, columns,
/// rows)`.
const MATVEC: [(&str, usize, usize); 4] = [
    ("blk.0.attn_qkv.weight", 2048, 8192),
    ("blk.0.ssm_out.weight", 4096, 2048),
    ("blk.0.ffn_down_shexp.weight", 512, 2048),
    ("blk.1.attn_q.weight", 2048, 8192),
];

/// The norm clause's tensors with the gain each loads under (the fixture's
/// own tensor, or none for a synthetic one). The second is also the residual
/// epilogue's and the fault inputs' tensor, the residual's the first three.
const NORM: [(&str, usize, usize, Option<&str>); 4] = [
    (
        "blk.0.attn_qkv.weight",
        2048,
        8192,
        Some("blk.0.attn_norm.weight"),
    ),
    ("blk.0.ssm_out.weight", 4096, 2048, None),
    ("blk.0.ffn_down_shexp.weight", 512, 2048, None),
    ("output.weight", 2048, 4096, Some("output_norm.weight")),
];

/// The segmented dispatches: each group's tensors with their rows.
const SEG: [&[(&str, usize)]; 2] = [
    &[
        ("blk.1.attn_q.weight", 8192),
        ("blk.1.attn_k.weight", 512),
        ("blk.1.attn_v.weight", 512),
    ],
    &[
        ("blk.0.attn_qkv.weight", 8192),
        ("blk.0.attn_gate.weight", 4096),
    ],
];

/// The gain the segments load under.
const SEG_GAIN: &str = "blk.1.attn_norm.weight";

/// The head's last rows: a count no threadgroup divides, deep in the
/// vocabulary's tensor.
const HEAD_TAIL: usize = 1153;

/// The store dot at the row's tensors and the head's tail rows.
fn matvec(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let mut parts = Vec::new();
    for (i, &(name, cols, rows)) in MATVEC.iter().enumerate() {
        let weights = fix.rows_as(quant, name, cols, rows)?;
        let x = oracle::activations(cols, 0x2A34 + i as u64, 1.0);
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
    let hidden = h.geom().hidden;
    let output_rows = fix.output_rows()?;
    let head = fix.rows_at_as(
        quant,
        "output.weight",
        hidden,
        output_rows - HEAD_TAIL,
        HEAD_TAIL,
    )?;
    let x = oracle::activations(hidden, 0x2A3A, 1.0);
    let worst = common::run_checked(
        h,
        &Launch {
            quant,
            rows: HEAD_TAIL,
            cols: hidden,
            weights: &head,
            x: &x,
            y0: &common::sentinel_rows(HEAD_TAIL),
            mode: ModeSpec::Store,
        },
    )
    .map_err(|e| {
        format!(
            "output.weight rows {}..{output_rows}: {e}",
            output_rows - HEAD_TAIL
        )
    })?;
    parts.push(format!(
        "output.weight tail {HEAD_TAIL}x{hidden} {worst:.3}"
    ));
    common::noted(parts)
}

/// Row counts no threadgroup divides: every row written, no word past the
/// last.
fn row_tail(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    let (name, cols) = ("blk.0.attn_qkv.weight", 2048);
    let all = fix.rows_as(quant, name, cols, 4095)?;
    let row_bytes = cols / quant.values_per_block() * quant.block_bytes();
    let x = oracle::activations(cols, 0x2A60, 1.0);
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

/// The norm on load and the residual epilogue, with the fault inputs the A4
/// clause runs: a NaN input, an infinite one, a zero input with a zero
/// epsilon, each raising the norm site and writing NaN rows.
fn norm(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let eps = fix.rms_eps()?;
    let mut parts = Vec::new();
    for (i, &(name, cols, rows, gain_name)) in NORM.iter().enumerate() {
        let gain = common::gain_for(fix, gain_name, cols, 0x2AA1 + i as u64)?;
        let weights = fix.rows_as(quant, name, cols, rows)?;
        let x = oracle::activations(cols, 0x2A40 + i as u64, 3.7);
        let worst = common::run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &weights,
                x: &x,
                y0: &common::sentinel_rows(rows),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            },
        )
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        parts.push(format!("{name} {rows}x{cols} {worst:.3}"));
    }

    let (name, cols, rows) = ("blk.0.ssm_out.weight", 4096, 2048);
    let weights = fix.rows_as(quant, name, cols, rows)?;
    let gain = common::gain_for(fix, None, cols, 0x2AB0)?;
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
                gain: &gain,
                eps,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("a zero input with eps {eps}: {e}"))?;

    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(cols, 1, 1.0), eps),
        ("an infinite input", oracle::activations(cols, 2, 1.0), eps),
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
                gain: &gain,
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
    for (i, &(name, cols, rows, _)) in NORM[..3].iter().enumerate() {
        let weights = fix.rows_as(quant, name, cols, rows)?;
        let x = oracle::activations(cols, 0x2A50 + i as u64, 1.0);
        let resid = oracle::activations(rows, 0x2BE5 + i as u64, 2.0);
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
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        parts.push(format!("resid {name} {rows}x{cols} {worst:.3}"));
    }
    common::noted(parts)
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone, in the A4 clause's layer shapes.
fn seg(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let eps = fix.rms_eps()?;
    let cols = 2048;
    let gain = fix.f32s(SEG_GAIN, cols)?;
    let mut parts = Vec::new();
    for (i, tensors) in SEG.iter().enumerate() {
        let x = oracle::activations(cols, 0x2E86 + i as u64, 1.0);
        let owned: Vec<Vec<u8>> = tensors
            .iter()
            .map(|&(name, n)| fix.rows_as(quant, name, cols, n))
            .collect::<Result<Vec<_>, String>>()?;
        let segments: Vec<(usize, &[u8])> = tensors
            .iter()
            .zip(&owned)
            .map(|(&(_, n), w)| (n, w.as_slice()))
            .collect();
        let rows: usize = tensors.iter().map(|&(_, n)| n).sum();
        let got = h.seg(&SegLaunch {
            quant,
            cols,
            segments: segments.clone(),
            x: &x,
            y0: &common::sentinel_rows(rows),
            gain: &gain,
            eps,
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
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            })?;
            for (k, (a, b)) in got[at..at + n].iter().zip(alone.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "{} segment {j} row {k}: the segment read {a:?}, the tensor alone read {b:?}",
                        tensors[j].0
                    ));
                }
            }
            at += n;
        }
        parts.push(format!("{} {} segments", tensors[0].0, tensors.len()));
    }
    common::noted(parts)
}

/// The head-normed residual mat-vec: the gated numerator times the scale of
/// the head each group sits in, against the f64 oracle, with the NaN partials
/// and numerator the A4 clause runs.
fn headnorm(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let eps = fix.rms_eps()?;
    let delta = &h.geom().delta;
    let cols = delta.head_cols * delta.value_heads;
    let weights = fix.rows_as(quant, "blk.0.ssm_out.weight", cols, 2048)?;
    let g = oracle::activations(cols, 0x2E88, 1.0);
    let partials = head_partials(delta);
    let x = common::head_scaled(
        &g,
        &partials,
        delta.value_heads,
        delta.head_cols,
        PARTIALS_PER_HEAD,
        eps,
    );
    let reference = oracle::reference(quant, &weights, 2048, cols, &x)?;
    let resid = oracle::activations(2048, 0x2E89, 2.0);
    let launch = |g: &[f32], partials: &[f32]| {
        h.headnorm(&HeadNormLaunch {
            quant,
            rows: 2048,
            cols,
            weights: &weights,
            g,
            partials,
            y0: &resid,
            eps,
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

/// The gate-up pair over the MLX stacks requantized as `quant`: the oracle
/// band, an id past the stack, and the norm's fault inputs.
fn pair(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let worst = check_pair_quant(h, fix, quant)?;
    let weights = clauses_experts::pair_weights_for(&h.geom().experts, fix, quant)?;
    clauses_experts::pair_faults(h, quant, &weights, &ffn_norm(fix)?)?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; ids -1 and the stack length raised expert_id, \
         NaN in that slot only; the norm's fault inputs raised norm_quant, NaN in all {} \
         slots' rows",
        h.geom().experts.slots
    )))
}

/// The folded down over the MLX stacks requantized as `quant`: the oracle band,
/// a row tail, and a bad id's NaN rows.
fn fold(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let worst = check_fold_quant(h, fix, quant, clauses_experts::FOLD_ROWS)?;
    let tail = check_fold_quant(h, fix, quant, clauses_experts::FOLD_TAIL_ROWS)?;
    let folded = clauses_experts::fold_weights_for(
        &h.geom().experts,
        fix,
        quant,
        clauses_experts::FOLD_ROWS,
    )?;
    fold_bad_id(h, &folded, quant)?;
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; tail {} rows: {tail}; a bad id writes NaN and \
         raises expert_id",
        clauses_experts::FOLD_TAIL_ROWS
    )))
}

/// The head over the MLX table requantized as `quant`: the logits against the
/// norm-on-load oracle band, the id against the host argmax of the read-back
/// logits, head and tail rows.
fn head(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let eps = fix.rms_eps()?;
    let hidden = h.geom().hidden;
    let gain = fix.f32s("output_norm.weight", hidden)?;
    let output_rows = fix.output_rows()?;
    let samples = [
        ("output.weight head", 0usize, 32usize, 0x2EAD_u64),
        ("output.weight tail", output_rows - 17, 17, 0x2EAE),
    ];
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, at, rows, seed) in samples {
        let weights = fix.rows_at_as(quant, "output.weight", hidden, at, rows)?;
        let x = oracle::activations(hidden, seed, 0.5);
        let out = clauses_head::run(
            h,
            clauses_head::HeadRun {
                quant,
                weights: &weights,
                x: &x,
                gain: &gain,
                rows,
                k: hidden,
                eps,
            },
        )?;
        if let Some(f) = h.fault()? {
            return Err(format!("{name} raised {f}"));
        }
        let xhat = oracle::rms_norm(&x, &gain, eps);
        let reference = oracle::reference(quant, &weights, rows, hidden, &xhat)?;
        let band = oracle::band(quant, hidden / quant.values_per_block(), Kind::Norm);
        let ratio = oracle::check(&out.logits, &reference, band, Kind::Norm, None)
            .map_err(|e| format!("{name}: {e}"))?;
        let (_, want) = clauses_head::host_argmax(&out.logits)?;
        if out.id != want {
            return Err(format!(
                "{name}: step id is {}, the logits select {want}",
                out.id
            ));
        }
        worst = worst.max(ratio);
        notes.push(format!("{name} {rows}x{hidden} {ratio:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "worst {worst:.3}; {}",
        notes.join(", ")
    )))
}

/// The rows of the embedding table the clause requantizes: enough
/// that the ids the clause embeds span the table's head and its end, without
/// quantizing the whole vocabulary at once.
const EMBED_ROWS: usize = 8192;

/// The embedding clause over the MLX table requantized as `quant`: the owner's
/// value decode bit-equal to the host reader, row by row, and the ids past
/// the table's rows.
fn embed(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let hidden = h.geom().hidden;
    let table = fix.rows_as(quant, "token_embd.weight", hidden, EMBED_ROWS)?;
    Ok(Verdict::Noted(clauses_embed::embed_table(
        h, quant, &table, EMBED_ROWS,
    )?))
}

// ------------------------------------------------------------------- the GEMMs

/// The dense GEMM's weights of `quant`, owned: the split's tensors requantized
/// as it, A4's own shapes — the three site tensors, the head's tail, the gate
/// projection's 16 rows (a quarter of a tile) and the segmented group's.
struct Gemm {
    quant: Quant,
    qkv: Vec<u8>,
    ssm_out: Vec<u8>,
    down: Vec<u8>,
    tail: Vec<u8>,
    gate: Vec<u8>,
    seg_qkv: Vec<u8>,
    seg_k: Vec<u8>,
    seg_tail: Vec<u8>,
}

impl Gemm {
    fn of(h: &Harness<'_>, mlx: &Fixture, quant: Quant) -> Result<Self, String> {
        let hidden = h.geom().hidden;
        let output_rows = mlx.output_rows()?;
        Ok(Self {
            quant,
            qkv: mlx.rows_as(quant, "blk.0.attn_qkv.weight", 2048, 8192)?,
            ssm_out: mlx.rows_as(quant, "blk.0.ssm_out.weight", 4096, 2048)?,
            down: mlx.rows_as(quant, "blk.0.ffn_down_shexp.weight", 512, 2048)?,
            tail: mlx.rows_at_as(
                quant,
                "output.weight",
                hidden,
                output_rows - HEAD_TAIL,
                HEAD_TAIL,
            )?,
            gate: mlx.rows_at_as(quant, "output.weight", hidden, 0, 16)?,
            seg_qkv: mlx.rows_as(quant, "blk.0.attn_qkv.weight", SEG_COLS, 8192)?,
            seg_k: mlx.rows_as(quant, "blk.1.attn_k.weight", SEG_COLS, 512)?,
            seg_tail: mlx.rows_at_as(quant, "blk.1.attn_q.weight", SEG_COLS, 100, 511)?,
        })
    }

    /// The store and resid cases: the three site tensors at the activation
    /// row counts that exercise the tile's edges, the head's tail, and the
    /// gate projection's 16 rows at two counts.
    fn cases<'a>(&'a self, h: &Harness<'_>) -> Vec<Case<'a>> {
        let hidden = h.geom().hidden;
        let quant = self.quant;
        let case = |label: String,
                    weights: &'a [u8],
                    rows: usize,
                    cols: usize,
                    acts: usize,
                    x_stride: usize,
                    y_stride: usize,
                    seed: u64| Case {
            label,
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
                "blk.0.attn_qkv.weight 8192x2048 r9".to_string(),
                &self.qkv,
                8192,
                2048,
                9,
                2048,
                8192,
                0x6EA1,
            ),
            case(
                "blk.0.ssm_out.weight 2048x4096 r33 strided".to_string(),
                &self.ssm_out,
                2048,
                4096,
                33,
                4096 + 16,
                2048 + 4,
                0x6EA2,
            ),
            case(
                "blk.0.ffn_down_shexp.weight 2048x512 r512".to_string(),
                &self.down,
                2048,
                512,
                512,
                512,
                2048,
                0x6EA3,
            ),
            case(
                format!("output.weight tail {HEAD_TAIL}x{hidden} r33"),
                &self.tail,
                HEAD_TAIL,
                hidden,
                33,
                hidden,
                HEAD_TAIL.div_ceil(4) * 4,
                0x6EA4,
            ),
            case(
                format!("output.weight head 16x{hidden} r33 stride 16"),
                &self.gate,
                16,
                hidden,
                33,
                hidden,
                16,
                0x6EA5,
            ),
            case(
                format!("output.weight head 16x{hidden} r9 stride 16"),
                &self.gate,
                16,
                hidden,
                9,
                hidden,
                16,
                0x6EA6,
            ),
        ]
    }

    /// The group of a segmented dispatch: two whole tensors, then the tail
    /// segment cut from the middle of a tensor.
    fn seg_group(&self) -> SegGroup<'_> {
        SegGroup {
            quant: self.quant,
            segments: vec![
                (8192, &self.seg_qkv[..]),
                (512, &self.seg_k[..]),
                (511, &self.seg_tail[..]),
            ],
            mode: GemmMode::Store,
        }
    }
}

/// The store dot at the row's tensors: every case within the f64 oracle's
/// band, both operand arms.
fn gemm_store(h: &Harness<'_>, mlx: &Fixture, quant: Quant) -> Clause {
    let w = Gemm::of(h, mlx, quant)?;
    run_cases(h, GemmMode::Store, &w.cases(h))
}

/// The residual epilogue at the same cases.
fn gemm_resid(h: &Harness<'_>, mlx: &Fixture, quant: Quant) -> Clause {
    let w = Gemm::of(h, mlx, quant)?;
    run_cases(h, GemmMode::Residual, &w.cases(h))
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone, both operand arms, both epilogues.
fn gemm_seg(h: &Harness<'_>, mlx: &Fixture, quant: Quant) -> Clause {
    let w = Gemm::of(h, mlx, quant)?;
    Ok(Verdict::Noted(format!(
        "every segment the same bits as its tensor alone: {}",
        seg_groups(h, &[w.seg_group()])?.join(", ")
    )))
}

/// The half-row GEMMs at the requantized cases: [`clauses_gemm::half_rows`].
fn gemm_half_rows(h: &Harness<'_>, mlx: &Fixture, quant: Quant) -> Clause {
    let w = Gemm::of(h, mlx, quant)?;
    clauses_gemm::half_rows_over(h, &w.cases(h), w.seg_group())
}

/// The grouped GEMM over synthetic stacks of `quant`: the pair and the down in both
/// operands within the band, runs ending inside a SIMD group's blocks, the
/// half range's fault, and a tile past the tile's slots
/// ([`clauses_route::grouped_synthetic`], the Q8_0 and Q5_K clauses' body).
fn grouped(h: &Harness<'_>, quant: Quant) -> Clause {
    super::clauses_route::grouped_synthetic(h, quant, 0x2A80)
}

/// The names of one type's clauses.
#[derive(Clone, Copy)]
struct Names {
    matvec: &'static str,
    row_tail: &'static str,
    norm: &'static str,
    seg: &'static str,
    headnorm: &'static str,
    pair: &'static str,
    fold: &'static str,
    head: &'static str,
    embed: &'static str,
    gemm_store: &'static str,
    gemm_resid: &'static str,
    gemm_seg: &'static str,
    gemm_half_rows: &'static str,
    grouped: &'static str,
    tensor_gemm: &'static str,
    tensor_gemm_seg: &'static str,
    tensor_grouped: &'static str,
}

/// The clause names of the type whose short name is `$p`.
macro_rules! names {
    ($p:literal) => {
        Names {
            matvec: concat!($p, "_matvec"),
            row_tail: concat!($p, "_row_tail"),
            norm: concat!($p, "_norm"),
            seg: concat!($p, "_seg"),
            headnorm: concat!($p, "_headnorm"),
            pair: concat!($p, "_pair"),
            fold: concat!($p, "_fold"),
            head: concat!($p, "_head"),
            embed: concat!($p, "_embed"),
            gemm_store: concat!("gemm_", $p, "_store"),
            gemm_resid: concat!("gemm_", $p, "_resid"),
            gemm_seg: concat!("gemm_", $p, "_seg"),
            gemm_half_rows: concat!("gemm_", $p, "_half_rows"),
            grouped: concat!("grouped_", $p),
            tensor_gemm: concat!("tensor_gemm_", $p),
            tensor_gemm_seg: concat!("tensor_gemm_", $p, "_seg"),
            tensor_grouped: concat!("tensor_grouped_", $p),
        }
    };
}

/// The types the split holds no typed site of ([`Fixture::requantizes`]),
/// each with its clauses' names: every one runs the same bodies.
const KINDS: [(Quant, Names); 3] = [
    (Quant::A8, names!("a8")),
    (Quant::A4H, names!("a4h")),
    (Quant::A8H, names!("a8h")),
];

/// `clause` at `quant`, over the MLX split.
fn run<'a>(
    h: &'a Harness<'a>,
    mlx: &'a Result<Fixture, String>,
    quant: Quant,
    clause: fn(&Harness<'_>, &Fixture, Quant) -> Clause,
) -> ClauseRun<'a> {
    with(h, mlx, move |h, fix| clause(h, fix, quant))
}

/// The requantized clauses of the whole row, in the token's order: the
/// mat-vec family's, then the prompt pass's follow in [`gemm_table`].
pub fn table<'a>(
    h: &'a Harness<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    KINDS
        .into_iter()
        .flat_map(|(q, n)| {
            [
                (n.matvec, run(h, mlx, q, matvec)),
                (n.row_tail, run(h, mlx, q, row_tail)),
                (n.norm, run(h, mlx, q, norm)),
                (n.seg, run(h, mlx, q, seg)),
                (n.headnorm, run(h, mlx, q, headnorm)),
                (n.pair, run(h, mlx, q, pair)),
                (n.fold, run(h, mlx, q, fold)),
                (n.head, run(h, mlx, q, head)),
                (n.embed, run(h, mlx, q, embed)),
            ]
        })
        .collect()
}

/// The requantized clauses of the prompt pass's GEMMs: the dense and
/// segmented GEMMs, the half-row entries, the grouped GEMM.
pub fn gemm_table<'a>(
    h: &'a Harness<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    KINDS
        .into_iter()
        .flat_map(|(q, n)| {
            [
                (n.gemm_store, run(h, mlx, q, gemm_store)),
                (n.gemm_resid, run(h, mlx, q, gemm_resid)),
                (n.gemm_seg, run(h, mlx, q, gemm_seg)),
                (n.gemm_half_rows, run(h, mlx, q, gemm_half_rows)),
                (n.grouped, Box::new(move || grouped(h, q)) as ClauseRun<'a>),
            ]
        })
        .collect()
}

/// The dense GEMM's tensor-op arms at the requantized cases.
fn tensor_gemm(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Clause {
    let w = Gemm::of(h, fix, quant)?;
    dense_arms(h, &w.cases(h))
}

/// The requantized clauses on the tensor-op kernels: the dense and segmented
/// GEMMs and the grouped GEMM, on `tensor` (or why this device has none).
pub fn tensor_table<'a>(
    tensor: &'a Result<Harness<'a>, String>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    KINDS
        .into_iter()
        .flat_map(|(q, n)| {
            let dense = move |clause: fn(&Harness<'_>, &Fixture, Quant) -> Clause| {
                Box::new(move || match (tensor, mlx) {
                    (Err(e), _) => Err(format!("the tensor-op GEMMs do not build here: {e}")),
                    (_, Err(e)) => Err(e.clone()),
                    (Ok(h), Ok(fix)) => {
                        h.clear_fault()?;
                        clause(h, fix, q)
                    }
                }) as ClauseRun<'a>
            };
            [
                (n.tensor_gemm, dense(tensor_gemm)),
                (n.tensor_gemm_seg, dense(gemm_seg)),
                (
                    n.tensor_grouped,
                    Box::new(move || match tensor {
                        Err(e) => Err(format!("the tensor-op GEMMs do not build here: {e}")),
                        Ok(h) => {
                            h.clear_fault()?;
                            grouped(h, q)
                        }
                    }) as ClauseRun<'a>,
                ),
            ]
        })
        .collect()
}
