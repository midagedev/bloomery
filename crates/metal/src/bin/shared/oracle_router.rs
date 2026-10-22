//! The router's oracle: an f64 dot of the normed input, and the band an f32
//! evaluation of that dot is held to.
//!
//! # The band
//!
//! PIN(2026-10-22): the model below, and no measured error, sets the band.
//!
//! Unit roundoff `U = 2^-24`. `G(n) = nU / (1 - nU)`. One `simd_sum` over 32
//! lanes is at most 31 adds, whatever its shape. The library `rsqrt` is taken
//! as `RSQRT_ULP` units in the last place of 1 (2 ulp = 4U), the same
//! unverified premise as the mat-vec oracle.
//!
//! **One split's dot.** A lane holds 32 products (8 float4 values): a gain
//! product and a weight product on each, then 31 adds. The SIMD group's sum
//! is 31 more, the scale multiplies once, and the last threadgroup adds split
//! 0 to split 1. `n_dot = 66`, `g = G(66)`.
//!
//! **The scale.** Each SIMD group sums 64 squares (16 float4 values): the
//! square, 63 adds, and 31 adds of the SIMD group, `n_S = 95`. The argument
//! of `rsqrt` is that sum divided by K (a power of two) plus eps, counted as
//! two more roundings the way the mat-vec oracle counts them, `t = G(97)`.
//! The scale is off by at most `(1 - t)^(-1/2) - 1` in the worse direction,
//! plus the library `rsqrt`. `theta_s` is that sum. The factor on
//! `sum_c |w_c xhat_c|` is `theta_s + g + theta_s * g`, with
//! `xhat = scale * gain * x` in f64. The absolute floor is one minimum
//! positive f32 per operation of the path (`97 + 66`). The written partial is
//! the f32 value the last threadgroup adds; that add is the last operation
//! of `n_dot`. The magnitude is multiplied by `1 + U`, as the mat-vec oracle
//! multiplies its own.
//!
//! **Ids.** The f64 ranking sorts the 256 expert logits descending and breaks
//! a tie toward the smaller id. The 8th-vs-9th margin is wider than the band
//! when it exceeds the sum of those two rows' logit widths: no value inside
//! its band can cross the cut. An internal adjacent margin inside its two
//! widths can still reorder the eight, so the id sequence is pinned only when
//! every adjacent margin of the top nine is wider than its band.
//!
//! **Weights.** `w_i = exp(l_i - l_max) / sum_j exp(l_j - l_max)` over the
//! selected logits, and the shared expert is
//! `sigmoid(l_256) = 1 / (1 + exp(-l_256))`. A logit movement of at most
//! `delta` (the widest selected width) moves a weight by at most
//! `0.5 * delta`, because `2 w (1 - w) <= 1/2`. The evaluation of that
//! formula, given the logits: the subtraction is a relative `U`, `exp` is at
//! most 4 ulp (`eta = 2^-21`) under the precise f32 functions the library
//! asks for, the sum of the eight is `G(7)`, and the division rounds once.
//! With `R = max(l_max - l_i) + 2 delta` and
//! `rel_e = (1 + eta) exp(U R) - 1`,
//! `rel_s = rel_e + G(7) (1 + rel_e)`,
//! `rel_w = (rel_e + rel_s) / (1 - rel_s) + U (1 + that)` when `rel_s < 1/2`,
//! else 1. The absolute band is `0.5 * delta + min(rel_w, 1)` plus the floor.
//! The shared sigmoid's Lipschitz factor is `1/4`. Its evaluation band is
//! derived in [`shared_band`], and is 1 when that expression is not a finite
//! bound.

use bloomery_metal::kernels::{EXPERTS, K, ROWS, TOP_K_MAX};

use super::oracle;

/// Unit roundoff of an f32 operation rounded to nearest.
const U: f64 = 1.0 / 16_777_216.0;

/// The most error of the library's `rsqrt`, in units in the last place.
const RSQRT_ULP: f64 = 2.0;

/// `exp` under the precise f32 functions, in ulp of the result.
const EXP_ULP: f64 = 4.0;

const LANE_PRODUCTS: usize = 32;
const LANE_ADDS: usize = LANE_PRODUCTS - 1;
const SIMD_SUM_DEPTH: usize = 31;
/// Gain product, weight product, the lane's adds, the SIMD group's adds, the
/// scale product, and the split add.
const N_DOT: usize = 1 + 1 + LANE_ADDS + SIMD_SUM_DEPTH + 1 + 1;
const SQUARES: usize = 64;
/// The square, the lane's adds of the squares, and the SIMD group's adds.
const N_S: usize = 1 + (SQUARES - 1) + SIMD_SUM_DEPTH;
/// The division by K and the epsilon, counted with the sum of squares.
const N_ARG: usize = N_S + 2;

const _: () = assert!(N_DOT == 66);
const _: () = assert!(N_S == 95);
const _: () = assert!(N_ARG == 97);
const _: () = assert!(K == 2048);
const _: () = assert!(ROWS == EXPERTS + 1);
const _: () = assert!(TOP_K_MAX == 8);

const fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// The relative factor of a logit and the operations that set its floor.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    factor: f64,
    ops: usize,
}

/// The band of one router logit.
#[must_use]
pub fn logit_band() -> Band {
    let t = gamma(N_ARG);
    let theta_s = (1.0 - t).powf(-0.5) - 1.0 + RSQRT_ULP * 2.0 * U;
    let g = gamma(N_DOT);
    Band {
        factor: theta_s + g + theta_s * g,
        ops: N_ARG + N_DOT,
    }
}

/// The absolute width of one logit: the factor times the sum of the terms'
/// magnitudes, the extra `1 + U`, and the floor.
#[must_use]
pub fn logit_width(band: Band, magnitude: f64) -> f64 {
    let floor = band.ops as f64 * f64::from(f32::MIN_POSITIVE);
    band.factor * magnitude * (1.0 + U) + floor
}

/// One router input, evaluated in f64: the logits and each row's width.
pub struct Step {
    /// The f64 logits, the shared expert last.
    pub logits: Vec<f64>,
    /// The absolute width of each logit.
    pub width: Vec<f64>,
}

/// `routed` is [`EXPERTS`] rows of [`K`], `shared` is [`K`], both f32.
pub fn step(
    name: &str,
    routed: &[f32],
    shared: &[f32],
    x: &[f32],
    gain: &[f32],
    eps: f32,
) -> Result<Step, String> {
    if routed.len() != EXPERTS * K || shared.len() != K || x.len() != K || gain.len() != K {
        return Err(format!(
            "{name}: the oracle was given {} routed values, {} shared, x {}, gain {}",
            routed.len(),
            shared.len(),
            x.len(),
            gain.len()
        ));
    }
    let xhat = oracle::rms_norm(x, gain, eps);
    let band = logit_band();
    let mut logits = Vec::with_capacity(ROWS);
    let mut width = Vec::with_capacity(ROWS);
    for row in 0..EXPERTS {
        let (dot, magnitude) = dot_row(&routed[row * K..(row + 1) * K], &xhat);
        logits.push(dot);
        width.push(logit_width(band, magnitude));
    }
    let (dot, magnitude) = dot_row(shared, &xhat);
    logits.push(dot);
    width.push(logit_width(band, magnitude));
    Ok(Step { logits, width })
}

fn dot_row(w: &[f32], xhat: &[f64]) -> (f64, f64) {
    let mut dot = 0.0;
    let mut magnitude = 0.0;
    for (&w, &x) in w.iter().zip(xhat) {
        let term = f64::from(w) * x;
        dot += term;
        magnitude += term.abs();
    }
    (dot, magnitude)
}

/// `got` against `step`, each logit inside its width. Returns the worst ratio
/// of the error to the width, or the first row outside it.
pub fn check_logits(got: &[f32], step: &Step) -> Result<f64, String> {
    if got.len() != step.logits.len() {
        return Err(format!(
            "{} logits read back, the oracle has {}",
            got.len(),
            step.logits.len()
        ));
    }
    let mut worst = 0.0f64;
    for (r, (&got, (&want, &width))) in got
        .iter()
        .zip(step.logits.iter().zip(&step.width))
        .enumerate()
    {
        let err = (f64::from(got) - want).abs();
        if !got.is_finite() || err > width {
            return Err(format!(
                "row {r}: got {got:e}, want {want:e}, error {err:e} over the band {width:e}"
            ));
        }
        worst = worst.max(err / width);
    }
    Ok(worst)
}

/// The f64 ranking of the 256 experts.
pub struct Ranking {
    /// The eight selected ids, best first, ties toward the smaller id.
    pub ids: Vec<usize>,
    /// The logit of the 8th minus the logit of the 9th.
    pub margin: f64,
    /// The sum of those two rows' widths.
    pub margin_band: f64,
    /// The 8th-vs-9th margin is wider than [`Ranking::margin_band`].
    pub cut_stable: bool,
    /// Every adjacent margin of the top nine is wider than its two widths,
    /// so the id sequence cannot reorder inside the band.
    pub sequence_stable: bool,
}

/// The ranking of `step`'s expert logits.
#[must_use]
pub fn rank(step: &Step) -> Ranking {
    let mut order: Vec<usize> = (0..EXPERTS).collect();
    order.sort_by(|&a, &b| match step.logits[b].partial_cmp(&step.logits[a]) {
        Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
        Some(ord) => ord,
    });
    let margin = step.logits[order[7]] - step.logits[order[8]];
    let margin_band = step.width[order[7]] + step.width[order[8]];
    let cut_stable = margin > margin_band;
    let mut sequence_stable = cut_stable;
    for i in 0..TOP_K_MAX {
        let gap = step.logits[order[i]] - step.logits[order[i + 1]];
        let band = step.width[order[i]] + step.width[order[i + 1]];
        if gap <= band {
            sequence_stable = false;
        }
    }
    Ranking {
        ids: order[..TOP_K_MAX].to_vec(),
        margin,
        margin_band,
        cut_stable,
        sequence_stable,
    }
}

/// The f64 weights of `ids` and the absolute band of each, shared expert last.
pub struct WeightOracle {
    /// The routed weights and then the shared expert's sigmoid.
    pub want: Vec<f64>,
    /// The absolute band of each.
    pub band: Vec<f64>,
}

/// The weights of `ids` (the routed selection, best first) from `step`.
pub fn weights(step: &Step, ids: &[usize]) -> Result<WeightOracle, String> {
    if ids.len() != TOP_K_MAX || ids.iter().any(|&id| id >= EXPERTS) {
        return Err(format!(
            "the weight oracle takes {TOP_K_MAX} expert ids, got {ids:?}"
        ));
    }
    let delta = ids.iter().map(|&id| step.width[id]).fold(0.0, f64::max);
    let l_max = ids
        .iter()
        .map(|&id| step.logits[id])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut reach = 0.0f64;
    for &id in ids {
        reach = reach.max(l_max - step.logits[id]);
    }
    reach += 2.0 * delta;
    let k = ids.len();
    let rel = weight_rel(reach, k);
    let floor = (k + 4) as f64 * f64::from(f32::MIN_POSITIVE);
    let abs = 0.5 * delta + rel + floor;
    let mut sum = 0.0;
    let mut term = Vec::with_capacity(k);
    for &id in ids {
        let e = (step.logits[id] - l_max).exp();
        term.push(e);
        sum += e;
    }
    let mut want: Vec<f64> = term.into_iter().map(|e| e / sum).collect();
    let mut band = vec![abs; k];
    want.push(sigmoid(step.logits[EXPERTS]));
    band.push(shared_band(step.logits[EXPERTS], step.width[EXPERTS]));
    Ok(WeightOracle { want, band })
}

/// `1 / (1 + exp(-l))`.
fn sigmoid(l: f64) -> f64 {
    1.0 / (1.0 + (-l).exp())
}

/// The relative evaluation error of the eight-exp weight, capped at 1.
fn weight_rel(reach: f64, k: usize) -> f64 {
    let eta = EXP_ULP * 2.0_f64.powi(-23);
    let rel_e = (1.0 + eta) * (U * reach).exp() - 1.0;
    let g = gamma(k.saturating_sub(1));
    let rel_s = rel_e + g * (1.0 + rel_e);
    if !rel_s.is_finite() || rel_s >= 0.5 {
        return 1.0;
    }
    let base = (rel_e + rel_s) / (1.0 - rel_s);
    let rel_w = base + U + U * base;
    if rel_w.is_finite() {
        rel_w.min(1.0)
    } else {
        1.0
    }
}

/// The shared expert's sigmoid band: the Lipschitz term plus the evaluation.
fn shared_band(l: f64, delta: f64) -> f64 {
    let eta = EXP_ULP * 2.0_f64.powi(-23);
    let lip = 0.25 * delta;
    let z_hi = (-l).exp() * delta.exp();
    let add_err = eta * z_hi + U * (1.0 + (1.0 + eta) * z_hi);
    if !add_err.is_finite() || !lip.is_finite() || add_err >= 1.0 {
        return 1.0;
    }
    let a_hat_lo = (1.0 - add_err).max(0.5);
    let div_err = add_err / a_hat_lo + U;
    if div_err.is_finite() {
        lip + div_err
    } else {
        1.0
    }
}

/// The host's selection from the kernel's own f32 logits: ascending scan,
/// strict `>`, so a tie keeps the smaller id. The shared id is [`EXPERTS`]
/// and is not ranked.
#[must_use]
pub fn select_f32(logits: &[f32], top_k: usize) -> Vec<u32> {
    let mut taken = [false; EXPERTS];
    let mut ids = Vec::with_capacity(top_k + 1);
    for _ in 0..top_k {
        let mut any = false;
        let mut best_v = 0.0f32;
        let mut best_i = 0u32;
        for i in 0..EXPERTS {
            if taken[i] {
                continue;
            }
            let v = logits[i];
            let better = v > best_v;
            if !any || better {
                any = true;
                best_v = v;
                best_i = u32::try_from(i).unwrap_or(0);
            }
        }
        taken[best_i as usize] = true;
        ids.push(best_i);
    }
    ids.push(u32::try_from(EXPERTS).unwrap_or(0));
    ids
}
