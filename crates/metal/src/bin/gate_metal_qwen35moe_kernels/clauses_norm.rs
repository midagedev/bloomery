//! The norm clauses: the norm on load and the residual epilogue of the
//! mat-vec, and the head-normed residual mat-vec of the gated delta net.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::kernels::{BLOCK_VALUES, HEAD_COLS, PARTIALS_PER_HEAD, Quant, VALUE_HEADS};

use super::common::{self, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::{Harness, HeadNormLaunch, Launch, ModeSpec};

pub fn norm_on_load(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let cases = [
        (
            common::Tensor::q4k("blk.0.attn_qkv.weight", 2048, 8192),
            Some("blk.0.attn_norm.weight"),
        ),
        (
            common::Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048),
            None,
        ),
        (
            common::Tensor::q4k("blk.0.ffn_down_shexp.weight", 512, 2048),
            None,
        ),
        (
            common::Tensor::q6k("blk.3.attn_v.weight", 2048, 512),
            Some("blk.0.attn_norm.weight"),
        ),
        (
            common::Tensor::q6k("output.weight", 2048, 4096),
            Some("output_norm.weight"),
        ),
    ];
    let mut parts = Vec::new();
    for (i, (t, gain_name)) in cases.iter().enumerate() {
        let gain = common::gain_for(fixture, *gain_name, t.cols, 0x6A1 + i as u64)?;
        // A mean square far from one, so the scale matters.
        let x = oracle::activations(t.cols, 0x4B40 + i as u64, 3.7);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
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

    // A zero input norms to zero with a positive epsilon: not a fault.
    let t = common::Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048);
    let gain = common::gain_for(fixture, None, t.cols, 0x6B0)?;
    common::run_checked(
        h,
        &Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fixture)?,
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

    // A mean square near the epsilon, so the epsilon is part of the scale.
    let small = oracle::activations(t.cols, 0x4B4F, 1.0e-3);
    let worst = common::run_checked(
        h,
        &Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fixture)?,
            x: &small,
            y0: &common::sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("an input whose mean square is near eps {eps}: {e}"))?;
    parts.push(format!("small input {worst:.3}"));

    // Input, epsilon or gain a norm cannot take: each raises the norm site
    // of the layer and writes NaN to every row. The infinite epsilon, the
    // NaN gain and the infinite gain run on both weight types, which share
    // the predicates.
    for (t, what, mut x, eps) in [
        (t, "a NaN input", oracle::activations(t.cols, 1, 1.0), eps),
        (
            t,
            "an infinite input",
            oracle::activations(t.cols, 2, 1.0),
            eps,
        ),
        (
            t,
            "a zero input and a zero epsilon",
            vec![0f32; t.cols],
            0.0,
        ),
        (
            t,
            "an infinite epsilon",
            oracle::activations(t.cols, 3, 1.0),
            f32::INFINITY,
        ),
        (
            common::Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
            "an infinite epsilon",
            oracle::activations(512, 3, 1.0),
            f32::INFINITY,
        ),
        (t, "a NaN gain", oracle::activations(t.cols, 4, 1.0), eps),
        (
            t,
            "an infinite gain",
            oracle::activations(t.cols, 5, 1.0),
            eps,
        ),
        (
            common::Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
            "a NaN gain",
            oracle::activations(512, 4, 1.0),
            eps,
        ),
        (
            common::Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
            "an infinite gain",
            oracle::activations(512, 5, 1.0),
            eps,
        ),
    ] {
        let mut gain = common::gain_for(fixture, None, t.cols, 0x6B0)?;
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[t.cols - 1] = f32::INFINITY,
            "a NaN gain" => gain[0] = f32::NAN,
            "an infinite gain" => gain[t.cols - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.matvec(&Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fixture)?,
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
    common::noted(parts)
}

pub fn residual_epilogue(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tensors = [
        common::Tensor::q4k("blk.0.attn_qkv.weight", 2048, 8192),
        common::Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048),
        common::Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
        common::Tensor::q6k("blk.2.attn_qkv.weight", 2048, 8192),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x4B50 + i as u64, 1.0);
        let resid = oracle::activations(t.rows, 0x7E5 + i as u64, 2.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &resid,
                mode: ModeSpec::Residual,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    common::noted(parts)
}

/// The head-normed residual mat-vec against the f64 oracle: the gated
/// numerator and the heads' partial sums are the inputs.
pub fn headnorm(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let t = common::Tensor::q4k("blk.0.ssm_out.weight", HEAD_COLS * VALUE_HEADS, 2048);
    let weights = t.bytes(fixture)?;
    let g = oracle::activations(HEAD_COLS * VALUE_HEADS, 0x5E8, 1.0);
    let partials = head_partials();
    let x = common::head_scaled(
        &g,
        &partials,
        VALUE_HEADS,
        HEAD_COLS,
        PARTIALS_PER_HEAD,
        eps,
    );
    let reference = oracle::reference(Quant::Q4K, weights, 2048, HEAD_COLS * VALUE_HEADS, &x)?;
    let resid = oracle::activations(2048, 0x5E9, 2.0);
    let got = h.headnorm(&HeadNormLaunch {
        rows: 2048,
        cols: HEAD_COLS * VALUE_HEADS,
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
        heads: VALUE_HEADS,
        cols_per_head: HEAD_COLS,
        partials_per_head: PARTIALS_PER_HEAD,
    });
    let band = oracle::band(Quant::Q4K, (HEAD_COLS * VALUE_HEADS) / BLOCK_VALUES, kind);
    let worst = oracle::check(&got, &reference, band, kind, Some(&resid))?;
    Ok(Verdict::Noted(format!(
        "the out projection's worst error over its band: {worst:.3}"
    )))
}

/// A head the norm cannot take: the gate's site is raised and the head is
/// NaN, not a zero head.
pub fn headnorm_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let t = common::Tensor::q4k("blk.0.ssm_out.weight", HEAD_COLS * VALUE_HEADS, 2048);
    let weights = t.bytes(fixture)?;
    let g = oracle::activations(HEAD_COLS * VALUE_HEADS, 0x5E8, 1.0);
    let partials = head_partials();
    let resid = oracle::activations(2048, 0x5E9, 2.0);
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
            rows: 2048,
            cols: HEAD_COLS * VALUE_HEADS,
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
    Ok(Verdict::Noted(
        "a partial and a gated numerator that are not finite each raised the gate site, every row NaN"
            .to_string(),
    ))
}

/// The heads' partial sums of the squares of their values: positive, and of
/// the size the values of a head have.
fn head_partials() -> Vec<f32> {
    (0..VALUE_HEADS * PARTIALS_PER_HEAD)
        .map(|i| 0.5 + (i % 7) as f32 * 0.25)
        .collect()
}
