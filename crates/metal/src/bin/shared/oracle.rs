//! The Metal kernels' oracle: `crates/gguf`'s dequantization of a weight's
//! blocks, an f64 dot on the host, and the band an f32 evaluation
//! of that dot is held to.
//!
//! # The band
//!
//! PIN(2026-10-22): the model below, and no measured error, sets the band.
//! (Re-derived the same day, before landing: the scale's `(1 + t)^(-1/2)` is
//! taken exactly, where it was taken to first order with a 1 % margin, and the
//! absolute floor counts its path's operations, where it was a count with no
//! derivation. The norm band narrowed by under half a percent.)
//!
//! Every f32 operation rounds once to nearest: `fl(a . b) = (a . b)(1 + d)`,
//! `|d| <= U = 2^-24`. A term of a sum that passes through `n` rounding
//! operations before the output carries a factor `(1 + d)^n`, bounded by
//! `1 + G(n)` with `G(n) = nU / (1 - nU)`, so the evaluated dot product holds
//! `|y_hat - y| <= G(n) * sum_c |t_c|` for the terms `t_c` of the evaluation
//! tree, `n` the longest path from a term to the output. Fusing a multiply and
//! an add (the compiler may) only removes a rounding, and the half-to-float
//! conversions, the integer masks and the products by a power of two are
//! exact.
//!
//! **Q6_K** (`matvec_q6k.metal`), one term `d * sc * q' * x`, `q'` an integer:
//! the product `x * q'` (1), the four-term sum (3 adds), the product by the
//! scale (1), the four-scale sum (3), the product by `d` (1), the lane's sum
//! over its `ceil(nb / 2)` blocks, and the SIMD group's sum over 32 lanes,
//! whose order the language does not fix, so the worst order counts: 31 adds.
//! `n = 9 + ceil(nb / 2) + 31`. The terms' magnitudes are `|w_c x_c|`.
//!
//! **Q4_K** (`matvec_q4k.metal`) factors a weight as `d sc q - dmin m`, two
//! terms that cancel inside a block, so the bound is on their magnitudes:
//! `M_c = d sc q + dmin m` (the same blocks with the sign of `dmin` flipped,
//! dequantized by the same reader), `sum_c M_c |x_c|`. The longest path is the
//! `dmin` side's: the eight-term `sumy` sum (7), the product by the scale (1),
//! the four-scale sum (3), the product by `dmin` (1), the subtraction (1), the
//! lane's sum over its `ceil(nb / 4)` blocks, and the 31 adds of the SIMD
//! group: `n = 13 + ceil(nb / 4) + 31`.
//!
//! **Norm on load** adds the gain product on the load (1) and the scale
//! product at the end (1), and the scale itself: `s = rsqrt(S / K + eps)`, `S`
//! the sum of squares in the kernel's own order (a square, two adds, the lane's
//! chain of `8 ceil(nb / 4)` adds for Q4_K or `4 ceil(nb / 2)` for Q6_K, the
//! 31 adds of the SIMD group), then the division and the addition. `s` is off
//! by at most `(1 - t)^(-1/2) - 1` for `t = G(nS + 2)` (the worse direction),
//! and by the library `rsqrt`'s own error, taken as `RSQRT_ULP` units in the
//! last place (2 ulp = 4U). The `RSQRT_ULP` premise is `[unverified]`: no source
//! for Metal's `rsqrt` bound was read, so it is the first term to question when
//! the norm clause misses. Its weight: at the widest norm the gate runs
//! (K = 4096, Q4_K), the term is 4U of about 88U in the norm's band factor, so
//! 4 ulp in place of 2 widens the norm band by about 5 % and moves no other
//! clause.
//! The band is on `sum_c M_c |x_hat_c|`, `x_hat = x * gain * s` in f64.
//!
//! **Floor.** A result below the smallest normal `f32::MIN_POSITIVE` may be
//! flushed to zero (the hardware's choice, not read here): an absolute error of
//! at most that per operation of the path, which the relative bound does not
//! cover. The gate's operands are not that small, so the floor only keeps a row
//! whose terms all vanish from being held to a band of zero.
//!
//! **Residual** adds the final `y + dot`: one rounding of the sum, `U |y|`.
//!
//! The oracle's own sums are f64 (`2^-53`), eight orders below `U`, and its
//! dequantized weights are the reader's f32 values, exact up to the one
//! rounding the reader itself makes.

use bloomery_metal::kernels::{BLOCK_VALUES, Quant};
use gguf::quant::{GgmlType, dequant_row};

/// Unit roundoff of an f32 operation rounded to nearest.
const U: f64 = 1.0 / 16_777_216.0;

/// The most error of the library's `rsqrt`, in units in the last place.
const RSQRT_ULP: f64 = 2.0;

/// The longest path through the SIMD group's sum: a binary tree over 32 lanes
/// is at most 31 adds deep, whatever its shape.
const SIMD_SUM_DEPTH: usize = 31;

fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// What the mat-vec does, as the band sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Store,
    Norm,
    Residual,
}

/// What a mat-vec's band is made of (the derivation above): the relative
/// factor of the sum of the terms' magnitudes, and the operations of the
/// longest path, which set the absolute floor.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    factor: f64,
    ops: usize,
}

/// The band of a mat-vec of `quant` over `nblocks` blocks of columns.
pub fn band(quant: Quant, nblocks: usize, kind: Kind) -> Band {
    let (lane_blocks, base) = match quant {
        Quant::Q4K => (nblocks.div_ceil(4), 13),
        Quant::Q6K => (nblocks.div_ceil(2), 9),
    };
    let n_store = base + lane_blocks + SIMD_SUM_DEPTH;
    match kind {
        Kind::Store | Kind::Residual => Band {
            factor: gamma(n_store),
            ops: n_store,
        },
        Kind::Norm => {
            let chain = match quant {
                Quant::Q4K => 8 * lane_blocks,
                Quant::Q6K => 4 * lane_blocks,
            };
            let n_s = 1 + 2 + chain + SIMD_SUM_DEPTH;
            // The scale's argument is off by at most `t` relative, so the
            // scale by at most (1 - t)^(-1/2) - 1 (the worse direction); the
            // library's rsqrt adds its own.
            let t = gamma(n_s + 2);
            let theta_s = (1.0 - t).powf(-0.5) - 1.0 + RSQRT_ULP * 2.0 * U;
            // The gain product joins the path, and the scale product ends it.
            let g = gamma(n_store + 2);
            Band {
                factor: theta_s + g + theta_s * g,
                ops: n_s + 2 + n_store + 2,
            }
        }
    }
}

/// One output row of the oracle: the f64 dot and the sum of the terms'
/// magnitudes the band is a multiple of.
pub struct Reference {
    pub y: Vec<f64>,
    pub magnitude: Vec<f64>,
}

fn ggml_type(quant: Quant) -> GgmlType {
    match quant {
        Quant::Q4K => GgmlType::Q4_K,
        Quant::Q6K => GgmlType::Q6_K,
    }
}

/// `block`'s bytes with `d` made non-negative and `dmin` non-positive: the
/// reader's dequantization of them is `d sc q + |dmin| m`, the magnitude of
/// the two terms of a Q4_K weight.
fn q4k_magnitude_blocks(row: &[u8]) -> Vec<u8> {
    const BLOCK: usize = Quant::Q4K.block_bytes();
    let mut out = row.to_vec();
    for block in out.as_chunks_mut::<BLOCK>().0 {
        block[1] &= 0x7F;
        block[3] |= 0x80;
    }
    out
}

/// The oracle for `rows` rows of `weights` against the input `x` (what the
/// kernel dots with: already normed when the kernel norms).
pub fn reference(
    quant: Quant,
    weights: &[u8],
    rows: usize,
    cols: usize,
    x: &[f64],
) -> Result<Reference, String> {
    let row_bytes = cols / BLOCK_VALUES * quant.block_bytes();
    if weights.len() != rows * row_bytes || x.len() != cols {
        return Err(format!(
            "oracle shape: {} weight bytes and {} inputs for {rows} x {cols}",
            weights.len(),
            x.len()
        ));
    }
    let ty = ggml_type(quant);
    let mut w = vec![0f32; cols];
    let mut m = vec![0f32; cols];
    let mut out = Reference {
        y: Vec::with_capacity(rows),
        magnitude: Vec::with_capacity(rows),
    };
    for row in weights.chunks_exact(row_bytes) {
        dequant_row(ty, row, &mut w).map_err(|e| format!("dequantizing a row: {e}"))?;
        let dot: f64 = w.iter().zip(x).map(|(&w, &x)| f64::from(w) * x).sum();
        let magnitude: f64 = match quant {
            Quant::Q6K => w
                .iter()
                .zip(x)
                .map(|(&w, &x)| f64::from(w).abs() * x.abs())
                .sum(),
            Quant::Q4K => {
                dequant_row(ty, &q4k_magnitude_blocks(row), &mut m)
                    .map_err(|e| format!("dequantizing a row's magnitudes: {e}"))?;
                m.iter().zip(x).map(|(&m, &x)| f64::from(m) * x.abs()).sum()
            }
        };
        out.y.push(dot);
        out.magnitude.push(magnitude);
    }
    Ok(out)
}

/// `x * gain / sqrt(mean(x^2) + eps)` in f64, the form of the host's
/// `rms_norm` (`bloomery_model::ops`, which does not build on a Mac): the scale
/// first, then the gain, then the value.
pub fn rms_norm(x: &[f32], gain: &[f32], eps: f32) -> Vec<f64> {
    let mean = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + f64::from(eps)).sqrt();
    x.iter()
        .zip(gain)
        .map(|(&x, &g)| scale * f64::from(g) * f64::from(x))
        .collect()
}

/// The part of the band a residual add contributes: `U |y|`.
fn residual_term(kind: Kind, total: f64) -> f64 {
    if kind == Kind::Residual {
        U * total.abs()
    } else {
        0.0
    }
}

/// `got` against `reference` row by row, each within the band; `base` is the
/// residual's input for the residual kind. Returns the worst ratio of the
/// error to the band, or the first row outside it.
pub fn check(
    got: &[f32],
    reference: &Reference,
    band: Band,
    kind: Kind,
    base: Option<&[f32]>,
) -> Result<f64, String> {
    if got.len() != reference.y.len() {
        return Err(format!(
            "{} rows read back, the oracle has {}",
            got.len(),
            reference.y.len()
        ));
    }
    let floor = band.ops as f64 * f64::from(f32::MIN_POSITIVE);
    let mut worst = 0f64;
    for (r, &got) in got.iter().enumerate() {
        let carried = base.map_or(0.0, |b| f64::from(b[r]));
        let want = carried + reference.y[r];
        let width =
            band.factor * reference.magnitude[r] * (1.0 + U) + residual_term(kind, want) + floor;
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

/// A deterministic stream of pseudo-random numbers (SplitMix64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    pub fn signed(&mut self) -> f32 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (2.0 * unit - 1.0) as f32
    }
}

/// `n` activations in `[-scale, scale)`, every 61st of them ten times larger.
pub fn activations(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let v = scale * rng.signed();
            if i % 61 == 7 { 10.0 * v } else { v }
        })
        .collect()
}

/// The dequantized values of a Q4_K table row, the embedding oracle.
pub fn q4k_row(table: &[u8], cols: usize, id: usize) -> Result<Vec<f32>, String> {
    let row_bytes = cols / BLOCK_VALUES * Quant::Q4K.block_bytes();
    let mut out = vec![0f32; cols];
    dequant_row(
        GgmlType::Q4_K,
        &table[id * row_bytes..(id + 1) * row_bytes],
        &mut out,
    )
    .map_err(|e| format!("dequantizing row {id}: {e}"))?;
    Ok(out)
}
