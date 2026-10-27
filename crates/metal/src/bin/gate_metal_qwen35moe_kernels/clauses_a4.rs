//! The A4 clauses: MLX's affine 4-bit decode (`quant.metal`'s `A4` owner)
//! against the f64 oracle, over the MLX-format fixture split. The Q4_K
//! clauses' own shapes and fault inputs, every new entry: the five mat-vec
//! modes, the pair, the fold, the head and the embedding. The mat-vec shapes
//! and the prompt pass's GEMM shapes (`clauses_gemm`) are a row's [`Sites`]:
//! the tensors of its family's trunk, at its geometry.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::kernels::{PARTIALS_PER_HEAD, Quant};

use super::clauses_embed;
use super::clauses_experts::{
    self, check_fold_quant, check_pair_quant, ffn_norm, fold_bad_id, fold_weights_for,
};
use super::clauses_head;
use super::clauses_norm::head_partials;
use super::common::{self, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::{Harness, HeadNormLaunch, Launch, ModeSpec};

/// One dense GEMM case at a row's site: a whole tensor against `acts`
/// activation rows, the activations' seed beside it.
pub struct GemmCase {
    pub(super) tensor: common::Tensor,
    pub(super) acts: usize,
    pub(super) seed: u64,
}

/// The segmented GEMM's A4 group at a row's site: whole tensors in segment
/// order, then one tail segment cut from a tensor, so the dispatch has a
/// segment no weight tile divides.
pub struct GemmSeg {
    /// The tensors taken whole, with their rows.
    pub(super) whole: &'static [(&'static str, usize)],
    /// The tail segment: the tensor, its first row and its rows.
    pub(super) tail: (&'static str, usize, usize),
}

/// The tensors the A4 clauses read at one row: the shapes of its family's
/// sites. The head and the embedding are the fixture's own.
pub struct Sites {
    /// The store dot's tensors.
    matvec: &'static [common::Tensor],
    /// The tensor the row counts are cut from.
    tail: common::Tensor,
    /// The norm clause's tensors with the gain each loads under (the fixture's
    /// own tensor, or none for a synthetic one). The second is also the
    /// residual epilogue's and the fault inputs' tensor, and the residual
    /// epilogue's tensors are the first three.
    norm: &'static [(common::Tensor, Option<&'static str>)],
    /// The segmented dispatches: each group's tensors with their rows.
    seg: &'static [&'static [(&'static str, usize)]],
    /// The gain the segments load under.
    seg_gain: &'static str,
    /// The dense prompt-pass GEMM's cases, before the head's tail.
    pub(super) gemm: &'static [GemmCase],
    /// The segmented prompt-pass GEMM's group.
    pub(super) gemm_seg: GemmSeg,
}

/// The Qwen3.6 row: the delta net's projections beside the gated attention
/// query and the shared expert's down.
pub const QWEN35_SITES: Sites = Sites {
    matvec: &[
        common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
        common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048),
        common::Tensor::a4("blk.0.ffn_down_shexp.weight", 512, 2048),
        common::Tensor::a4("blk.1.attn_q.weight", 2048, 8192),
    ],
    tail: common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
    norm: &[
        (
            common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
            Some("blk.0.attn_norm.weight"),
        ),
        (common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048), None),
        (
            common::Tensor::a4("blk.0.ffn_down_shexp.weight", 512, 2048),
            None,
        ),
        (
            common::Tensor::a4("output.weight", 2048, 4096),
            Some("output_norm.weight"),
        ),
    ],
    seg: &[
        &[
            ("blk.1.attn_q.weight", 8192),
            ("blk.1.attn_k.weight", 512),
            ("blk.1.attn_v.weight", 512),
        ],
        &[
            ("blk.0.attn_qkv.weight", 8192),
            ("blk.0.attn_gate.weight", 4096),
        ],
    ],
    seg_gain: "blk.1.attn_norm.weight",
    gemm: &[
        GemmCase {
            tensor: common::Tensor::a4("blk.0.attn_qkv.weight", 2048, 8192),
            acts: 9,
            seed: 0x6E51,
        },
        GemmCase {
            tensor: common::Tensor::a4("blk.0.ssm_out.weight", 4096, 2048),
            acts: 33,
            seed: 0x6E52,
        },
        GemmCase {
            tensor: common::Tensor::a4("blk.0.ffn_down_shexp.weight", 512, 2048),
            acts: 512,
            seed: 0x6E53,
        },
    ],
    gemm_seg: GemmSeg {
        whole: &[
            ("blk.0.attn_qkv.weight", 8192),
            ("blk.1.attn_k.weight", 512),
        ],
        tail: ("blk.1.attn_q.weight", 100, 511),
    },
};

/// The Qwen3-30B row: the plain attention query of 32 heads of 128, the key
/// and value of 4, and the output projection over the 4096 values they
/// make.
pub const QWEN3_SITES: Sites = Sites {
    matvec: &[
        common::Tensor::a4("blk.0.attn_q.weight", 2048, 4096),
        common::Tensor::a4("blk.0.attn_output.weight", 4096, 2048),
        common::Tensor::a4("blk.0.attn_k.weight", 2048, 512),
        common::Tensor::a4("blk.1.attn_v.weight", 2048, 512),
    ],
    tail: common::Tensor::a4("blk.0.attn_q.weight", 2048, 4096),
    norm: &[
        (
            common::Tensor::a4("blk.0.attn_q.weight", 2048, 4096),
            Some("blk.0.attn_norm.weight"),
        ),
        (
            common::Tensor::a4("blk.0.attn_output.weight", 4096, 2048),
            None,
        ),
        (common::Tensor::a4("blk.0.attn_k.weight", 2048, 512), None),
        (
            common::Tensor::a4("output.weight", 2048, 4096),
            Some("output_norm.weight"),
        ),
    ],
    seg: &[&[
        ("blk.1.attn_q.weight", 4096),
        ("blk.1.attn_k.weight", 512),
        ("blk.1.attn_v.weight", 512),
    ]],
    seg_gain: "blk.1.attn_norm.weight",
    gemm: &[
        GemmCase {
            tensor: common::Tensor::a4("blk.0.attn_q.weight", 2048, 4096),
            acts: 9,
            seed: 0x6E55,
        },
        GemmCase {
            tensor: common::Tensor::a4("blk.0.attn_output.weight", 4096, 2048),
            acts: 33,
            seed: 0x6E56,
        },
        GemmCase {
            tensor: common::Tensor::a4("blk.0.attn_k.weight", 2048, 512),
            acts: 512,
            seed: 0x6E57,
        },
    ],
    gemm_seg: GemmSeg {
        whole: &[
            ("blk.1.attn_q.weight", 4096),
            ("blk.1.attn_k.weight", 512),
            ("blk.1.attn_v.weight", 512),
        ],
        tail: ("blk.1.attn_q.weight", 100, 511),
    },
};

/// The store dot at the row's sites and the head's tail rows.
pub fn a4_matvec(h: &Harness<'_>, fix: &Fixture, sites: &Sites) -> Clause {
    let mut parts = Vec::new();
    for (i, t) in sites.matvec.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x1A34 + i as u64, 1.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fix)?,
                x: &x,
                y0: &common::sentinel_rows(t.rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    // The head's last rows: a count no threadgroup divides, deep in the
    // vocabulary's tensor, in the Q6_K clause's shape.
    let tail_rows = 1153;
    let hidden = h.geom().hidden;
    let output_rows = fix.output_rows()?;
    let head = fix.rows_at(
        "output.weight",
        gguf::quant::GgmlType::MLX_A4,
        hidden,
        output_rows - tail_rows,
        tail_rows,
    )?;
    let x = oracle::activations(hidden, 0x1A3A, 1.0);
    let worst = common::run_checked(
        h,
        &Launch {
            quant: Quant::A4,
            rows: tail_rows,
            cols: hidden,
            weights: head,
            x: &x,
            y0: &common::sentinel_rows(tail_rows),
            mode: ModeSpec::Store,
        },
    )
    .map_err(|e| {
        format!(
            "output.weight rows {}..{output_rows}: {e}",
            output_rows - tail_rows
        )
    })?;
    parts.push(format!(
        "output.weight tail {tail_rows}x{hidden} {worst:.3}"
    ));
    common::noted(parts)
}

/// Row counts no threadgroup divides: every row written, no word past the
/// last, in the Q4_K clause's counts.
pub fn a4_row_tail(h: &Harness<'_>, fix: &Fixture, sites: &Sites) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    let t = sites.tail;
    let all = t.bytes(fix)?;
    let row_bytes = t.cols / Quant::A4.values_per_block() * Quant::A4.block_bytes();
    let x = oracle::activations(t.cols, 0x1A60, 1.0);
    for rows in counts {
        common::run_checked(
            h,
            &Launch {
                quant: Quant::A4,
                rows,
                cols: t.cols,
                weights: &all[..rows * row_bytes],
                x: &x,
                y0: &common::sentinel_rows(rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{} rows of {}: {e}", rows, t.label()))?;
    }
    Ok(Verdict::Noted(format!(
        "{} row counts, every row written, no word past the last",
        counts.len()
    )))
}

/// The norm on load and the residual epilogue, with the fault inputs the
/// Q4_K clause runs: a NaN input, an infinite one, a zero input with a zero
/// epsilon, each raising the norm site and writing NaN rows.
pub fn a4_norm(h: &Harness<'_>, fix: &Fixture, sites: &Sites) -> Clause {
    let eps = fix.rms_eps()?;
    let mut parts = Vec::new();
    for (i, (t, gain_name)) in sites.norm.iter().enumerate() {
        let gain = common::gain_for(fix, *gain_name, t.cols, 0x1AA1 + i as u64)?;
        let x = oracle::activations(t.cols, 0x1A40 + i as u64, 3.7);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fix)?,
                x: &x,
                y0: &common::sentinel_rows(t.rows),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }

    let t = sites.norm[1].0;
    let gain = common::gain_for(fix, None, t.cols, 0x1AB0)?;
    // A zero input norms to zero with a positive epsilon: not a fault.
    common::run_checked(
        h,
        &Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fix)?,
            x: &vec![0f32; t.cols],
            y0: &common::sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("a zero input with eps {eps}: {e}"))?;

    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(t.cols, 1, 1.0), eps),
        (
            "an infinite input",
            oracle::activations(t.cols, 2, 1.0),
            eps,
        ),
        ("a zero input and a zero epsilon", vec![0f32; t.cols], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[t.cols - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.matvec(&Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fix)?,
            x: &x,
            y0: &common::sentinel_rows(t.rows),
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
    for (i, (t, _)) in sites.norm[..3].iter().enumerate() {
        let x = oracle::activations(t.cols, 0x1A50 + i as u64, 1.0);
        let resid = oracle::activations(t.rows, 0x1BE5 + i as u64, 2.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fix)?,
                x: &x,
                y0: &resid,
                mode: ModeSpec::Residual,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("resid {} {worst:.3}", t.label()));
    }
    common::noted(parts)
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone, in the Q4_K clause's layer shapes.
pub fn a4_seg(h: &Harness<'_>, fix: &Fixture, sites: &Sites) -> Clause {
    let eps = fix.rms_eps()?;
    let cols = 2048;
    let ty = gguf::quant::GgmlType::MLX_A4;
    let gain = fix.f32s(sites.seg_gain, cols)?;
    let mut parts = Vec::new();
    for (i, tensors) in sites.seg.iter().enumerate() {
        let x = oracle::activations(cols, 0x1E86 + i as u64, 1.0);
        let mut segments: Vec<(usize, &[u8])> = Vec::new();
        let mut rows = 0usize;
        for (name, n) in *tensors {
            segments.push((*n, fix.rows(name, ty, cols, *n)?));
            rows += n;
        }
        let got = h.seg(&super::SegLaunch {
            quant: Quant::A4,
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
                quant: Quant::A4,
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
/// and numerator the Q4_K clause runs.
pub fn a4_headnorm(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let eps = fix.rms_eps()?;
    let delta = &h.geom().delta;
    let cols = delta.head_cols * delta.value_heads;
    let t = common::Tensor::a4("blk.0.ssm_out.weight", cols, 2048);
    let weights = t.bytes(fix)?;
    let g = oracle::activations(cols, 0x1E88, 1.0);
    let partials = head_partials(delta);
    let x = common::head_scaled(
        &g,
        &partials,
        delta.value_heads,
        delta.head_cols,
        PARTIALS_PER_HEAD,
        eps,
    );
    let reference = oracle::reference(Quant::A4, weights, 2048, cols, &x)?;
    let resid = oracle::activations(2048, 0x1E89, 2.0);
    let got = h.headnorm(&HeadNormLaunch {
        quant: Quant::A4,
        rows: 2048,
        cols,
        weights,
        g: &g,
        partials: &partials,
        y0: &resid,
        eps,
        layer: LAYER,
    })?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the launch raised {fault}"));
    }
    let kind = Kind::HeadNorm(oracle::HeadGeometry {
        heads: delta.value_heads,
        cols_per_head: delta.head_cols,
        partials_per_head: PARTIALS_PER_HEAD,
    });
    let band = oracle::band(Quant::A4, cols / Quant::A4.values_per_block(), kind);
    let worst = oracle::check(&got, &reference, band, kind, Some(&resid))?;
    let mut noted = format!("worst error over its band: {worst:.3}");

    // A head the norm cannot take: the gate's site is raised and the head is
    // NaN, not a zero head.
    let mut bad_partials = partials.clone();
    bad_partials[5 * PARTIALS_PER_HEAD + 3] = f32::INFINITY;
    let mut bad_g = g.clone();
    bad_g[17] = f32::NAN;
    for (what, g, partials) in [
        ("a partial that is not finite", &g, &bad_partials),
        ("a gated numerator that is not finite", &bad_g, &partials),
    ] {
        h.clear_fault()?;
        let got = h.headnorm(&HeadNormLaunch {
            quant: Quant::A4,
            rows: 2048,
            cols,
            weights,
            g,
            partials,
            y0: &resid,
            eps,
            layer: LAYER,
        })?;
        common::expect_fault(h, what, FaultSite::LinearGate)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;
    noted.push_str(
        "; a partial and a gated numerator that are not finite each raised the gate site, every row NaN",
    );
    Ok(Verdict::Noted(noted))
}

/// The gate-up pair over the MLX stacks: the oracle band, an id past the
/// stack, and the norm's fault inputs.
pub fn a4_pair(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let worst = check_pair_quant(h, fix, Quant::A4)?;
    let weights = clauses_experts::pair_weights_for(&h.geom().experts, fix, Quant::A4)?;
    clauses_experts::pair_faults(h, Quant::A4, &weights, &ffn_norm(fix)?)?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; ids -1 and the stack length raised expert_id, \
         NaN in that slot only; the norm's fault inputs raised norm_quant, NaN in all {} \
         slots' rows",
        h.geom().experts.slots
    )))
}

/// The folded down over the MLX stacks: the oracle band, a row tail, and a
/// bad id's NaN rows.
pub fn a4_fold(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let worst = check_fold_quant(h, fix, Quant::A4, 2048)?;
    let tail = check_fold_quant(h, fix, Quant::A4, 2045)?;
    let folded = fold_weights_for(&h.geom().experts, fix, Quant::A4, 2048)?;
    fold_bad_id(h, &folded, Quant::A4)?;
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {worst}; tail 2045 rows: {tail}; a bad id writes NaN and \
         raises expert_id"
    )))
}

/// The head over the MLX table: the logits against the norm-on-load oracle
/// band, the id against the host argmax of the read-back logits, head and
/// tail rows.
pub fn a4_head(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let eps = fix.rms_eps()?;
    let hidden = h.geom().hidden;
    let gain = fix.f32s("output_norm.weight", hidden)?;
    let output_rows = fix.output_rows()?;
    let samples = [
        ("output.weight head", 0usize, 32usize, 0x1EAD_u64),
        ("output.weight tail", output_rows - 17, 17, 0x1EAE),
    ];
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, at, rows, seed) in samples {
        let weights = fix.rows_at(
            "output.weight",
            gguf::quant::GgmlType::MLX_A4,
            hidden,
            at,
            rows,
        )?;
        let x = oracle::activations(hidden, seed, 0.5);
        let out = clauses_head::run(
            h,
            clauses_head::HeadRun {
                quant: Quant::A4,
                weights,
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
        if out.ticket != 0 {
            return Err(format!(
                "{name}: the ticket counter is {}, not 0",
                out.ticket
            ));
        }
        let xhat = oracle::rms_norm(&x, &gain, eps);
        let reference = oracle::reference(Quant::A4, weights, rows, hidden, &xhat)?;
        let band = oracle::band(Quant::A4, hidden / Quant::A4.values_per_block(), Kind::Norm);
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

/// The embedding clause over the MLX table: the fixture's table is MLX_A4,
/// so `embed` holds the owner's value decode bit-equal to the host reader,
/// row by row, and the ids past the vocabulary.
pub fn a4_embed(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let (_, _, ty) = fix.table("token_embd.weight", h.geom().hidden)?;
    if ty != gguf::quant::GgmlType::MLX_A4 {
        return Err(format!("token_embd.weight is {ty:?}, not MLX_A4"));
    }
    clauses_embed::embed(h, fix)
}

/// `clause` over the MLX split, or the reason it could not be opened: a clause
/// run without the split is red, not skipped.
pub(super) fn with<'a>(
    h: &'a Harness<'a>,
    mlx: &'a Result<Fixture, String>,
    clause: impl Fn(&Harness<'_>, &Fixture) -> Clause + 'a,
) -> super::ClauseRun<'a> {
    Box::new(move || match mlx {
        Ok(fix) => clause(h, fix),
        Err(e) => Err(e.clone()),
    })
}

/// The clause table of the whole row: one clause per entry, in the token's order.
pub fn table<'a>(
    h: &'a super::Harness<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    let sites = &QWEN35_SITES;
    vec![
        (
            "a4_matvec",
            with(h, mlx, move |h, f| a4_matvec(h, f, sites)),
        ),
        (
            "a4_row_tail",
            with(h, mlx, move |h, f| a4_row_tail(h, f, sites)),
        ),
        ("a4_norm", with(h, mlx, move |h, f| a4_norm(h, f, sites))),
        ("a4_seg", with(h, mlx, move |h, f| a4_seg(h, f, sites))),
        ("a4_headnorm", with(h, mlx, a4_headnorm)),
        ("a4_pair", with(h, mlx, a4_pair)),
        ("a4_fold", with(h, mlx, a4_fold)),
        ("a4_head", with(h, mlx, a4_head)),
        ("a4_embed", with(h, mlx, a4_embed)),
    ]
}

/// The clauses of a row of no delta net: [`table`]'s but the head-normed
/// residual, at that row's sites.
pub fn shaped<'a>(
    h: &'a super::Harness<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    let sites = &QWEN3_SITES;
    vec![
        (
            "a4_matvec",
            with(h, mlx, move |h, f| a4_matvec(h, f, sites)),
        ),
        (
            "a4_row_tail",
            with(h, mlx, move |h, f| a4_row_tail(h, f, sites)),
        ),
        ("a4_norm", with(h, mlx, move |h, f| a4_norm(h, f, sites))),
        ("a4_seg", with(h, mlx, move |h, f| a4_seg(h, f, sites))),
        ("a4_pair", with(h, mlx, a4_pair)),
        ("a4_fold", with(h, mlx, a4_fold)),
        ("a4_head", with(h, mlx, a4_head)),
        ("a4_embed", with(h, mlx, a4_embed)),
    ]
}
