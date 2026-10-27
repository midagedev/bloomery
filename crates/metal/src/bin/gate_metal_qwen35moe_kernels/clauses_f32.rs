//! The F32 clauses: f32 weight rows (quant.metal's `F32`), the type of the
//! delta net's `ssm_alpha` and `ssm_beta`, whose one launch is the segmented
//! mat-vec. Segments of synthetic rows at the layer's shapes run against the
//! f64 oracle in the norm-on-load form the P1 launches, and the launches that
//! have no f32 entry refuse it by name (`site_types` holds the predicates).

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::kernels::Quant;

use super::common::{self, LAYER};
use super::oracle::{self, Kind};
use super::synthetic;
use super::{Harness, SegLaunch, clauses_synth};

/// The epsilon every launch of the clause takes.
const EPS: f32 = 1e-6;

/// The input columns of the delta net's alpha and beta: the hidden width.
const COLS: usize = 2048;

/// A segmented dispatch of f32 rows against the oracle: the segment lists a
/// layer's P1 holds (the delta net's two value-head projections, and the most
/// segments a dispatch takes), a row count no threadgroup divides, and the
/// norm's fault inputs.
pub fn f32_seg(h: &Harness<'_>) -> Clause {
    let lists: [(&str, &[usize]); 4] = [
        ("beta and alpha", &[32, 32]),
        ("one segment", &[5]),
        ("four segments", &[32, 7, 33, 1]),
        ("a long tail", &[4093]),
    ];
    let band = oracle::band(Quant::F32, COLS / Quant::F32.values_per_block(), Kind::Norm);
    let mut parts = Vec::new();
    for (i, (name, rows)) in lists.iter().enumerate() {
        let seed = 0xF320 + 16 * i as u64;
        let tensors: Vec<Vec<u8>> = rows
            .iter()
            .enumerate()
            .map(|(j, &n)| synthetic::matrix(Quant::F32, n, COLS, seed + j as u64))
            .collect();
        let gain = clauses_synth::gain(COLS, seed + 8);
        let x = oracle::activations(COLS, seed + 9, 3.7);
        let total: usize = rows.iter().sum();
        h.clear_fault()?;
        let got = h.seg(&SegLaunch {
            quant: Quant::F32,
            cols: COLS,
            segments: rows
                .iter()
                .zip(&tensors)
                .map(|(&n, bytes)| (n, bytes.as_slice()))
                .collect(),
            x: &x,
            y0: &common::sentinel_rows(total),
            gain: &gain,
            eps: EPS,
            layer: LAYER,
        })?;
        common::expect_clean(h, name)?;
        let effective = oracle::rms_norm(&x, &gain, EPS);
        let mut at = 0usize;
        let mut worst = 0f64;
        for (j, (&n, bytes)) in rows.iter().zip(&tensors).enumerate() {
            let reference = oracle::reference(Quant::F32, bytes, n, COLS, &effective)?;
            let ratio = oracle::check(&got[at..at + n], &reference, band, Kind::Norm, None)
                .map_err(|e| format!("{name} segment {j}: {e}"))?;
            worst = worst.max(ratio);
            at += n;
        }
        parts.push(format!("{name} {worst:.3}"));
    }

    // The norm's fault inputs: the norm site and NaN rows, as every type's.
    let rows = [32usize, 32];
    let tensors: Vec<Vec<u8>> = rows
        .iter()
        .enumerate()
        .map(|(j, &n)| synthetic::matrix(Quant::F32, n, COLS, 0xF3F0 + j as u64))
        .collect();
    let g = clauses_synth::gain(COLS, 0xF3F8);
    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(COLS, 1, 1.0), EPS),
        ("an infinite input", oracle::activations(COLS, 2, 1.0), EPS),
        ("a zero input and a zero epsilon", vec![0f32; COLS], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[COLS - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.seg(&SegLaunch {
            quant: Quant::F32,
            cols: COLS,
            segments: rows
                .iter()
                .zip(&tensors)
                .map(|(&n, bytes)| (n, bytes.as_slice()))
                .collect(),
            x: &x,
            y0: &common::sentinel_rows(64),
            gain: &g,
            eps,
            layer: LAYER,
        })?;
        common::expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}; the norm's fault inputs are the norm site and NaN rows",
        parts.join(", ")
    )))
}
