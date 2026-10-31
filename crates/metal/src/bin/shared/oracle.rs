//! The Metal kernels' oracle: `crates/gguf`'s dequantization of a weight's
//! blocks, an f64 dot on the host, and the band an f32 evaluation
//! of that dot is held to.
//!
//! # The band
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
//! **A4** (MLX's affine 4-bit, quant.metal's `A4`) factors a weight as
//! `s q + b`, two terms that cancel inside a group, so the bound is on their
//! magnitudes: `M_c = |s| q + |b|` (the same groups with both sign bits
//! cleared, dequantized by the same reader), `sum_c M_c |x_c|`. The longest
//! path is the code side's: the product (1), the four-term chain of its
//! accumulator (3), the three combine adds (each an exact power-of-two
//! product and one rounding), the product by the scale (1) and the bias
//! side's add (1) — the bias side itself is shorter (the load's `sumy`
//! chain of 4 adds per block, its grouping's 2, the product, the add) — the
//! lane's sum over its `ceil(nb / 8)` blocks, and the 31 adds of the SIMD
//! group: `n = 9 + ceil(nb / 8) + 31`.
//!
//! **A8** (MLX's affine 8-bit, quant.metal's `A8`) factors a weight as A4's
//! `s q + b`, so the bound is on A4's magnitudes (`M_c = |s| q + |b|`, the
//! groups with both sign bits cleared, dequantized by the same reader). The
//! longest path is A4's own count over its own tree: the product (1), the
//! four-term chain of its accumulator (3), the three combine adds (A8's are
//! plain adds, the codes unpacked to their values; no power-of-two product,
//! and none is needed), the product by the scale (1) and the bias side's add
//! (1), the lane's sum over its `ceil(nb / 8)` blocks, and the 31 adds of
//! the SIMD group: `n = 9 + ceil(nb / 8) + 31`.
//!
//! **A4H** and **A8H** are those owners over F16 scales and biases: the pair
//! widens exactly (an F16's significand is 11 bits, a product with a code of
//! at most 8 bits still fits f32), so the evaluation tree, the magnitudes (the
//! sign is the top bit of an F16 as it is of a BF16) and the count are A4's
//! and A8's.
//!
//! **Q8_0** (quant.metal's `Q8_0`), one term `d * q * x`, `q` an integer: the
//! product `q * x` (1), the eight-term chain of the block's code dot (7: its
//! first add is to zero, exact), the product by `d` (1), the lane's sum over
//! its `ceil(nb / 8)` blocks, and the 31 adds of the SIMD group:
//! `n = 9 + ceil(nb / 8) + 31`. The terms' magnitudes are `|w_c x_c|`.
//!
//! **Q5_K** (quant.metal's `Q5K`) factors a weight as Q4_K's `d sc q - dmin m`
//! with a five-bit `q`, so the bound is on Q4_K's magnitudes, the same sign
//! flip at the same offsets: `M_c = d sc q + dmin m`, `sum_c M_c |x_c|`. A
//! code's fifth bit enters as a sum of the activations whose bit is set,
//! times 16 (exact), so the terms of `d sc q x` are `q_lo x` and `16 x` and
//! their magnitudes sum to `q |x|`. The longest path is the `d` side's: the
//! product (1), the eight-term chain of the nibble sum (7), the combine with
//! the fifth bits' sum (1; the divide and multiply by 16 are exact), the
//! product by the scale (1), the four-scale sum (3), the product by `d` (1)
//! and the subtraction (1) — the `dmin` side is Q4_K's 13 — the lane's sum
//! over its `ceil(nb / 4)` blocks, and the 31 adds of the SIMD group:
//! `n = 15 + ceil(nb / 4) + 31`. The head-normed form scales each half once
//! and adds the halves: the two-scale sum (1), the head scale (1) and that
//! add (1) in place of the four-scale sum (3), so its path is the store's
//! plus the head scale's product, within `n + 1` as Q4_K's.
//!
//! **F32** (quant.metal's `F32`), one term `w * x`: the product (1), the two
//! pair adds of a quarter's four products (2), the lane's sum over its
//! `ceil(nb / 8)` blocks of 16 values, and the 31 adds of the SIMD group:
//! `n = 3 + ceil(nb / 8) + 31`. The terms' magnitudes are `|w_c x_c|`.
//!
//! **Norm on load** adds the gain product on the load (1) and the scale
//! product at the end (1), and the scale itself: `s = rsqrt(S / K + eps)`, `S`
//! the sum of squares in the kernel's own order (a square, two adds, the lane's
//! chain of `8 ceil(nb / 4)` adds for Q4_K and Q5_K (one load), `4 ceil(nb /
//! 2)` for Q6_K, `4 ceil(nb / 8)` for A4 and A8, `2 ceil(nb / 8)` for Q8_0 or `ceil(nb / 8)` for F32, the 31
//! adds of the SIMD group), then the division and the addition. `s` is off
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
//! **Head norm on load** (`q4k_matvec_headnorm_resid`): the input is the
//! gated numerator `g` times the norm scale of the head each column belongs
//! to, and the scale comes from the head's partial sums of the squares of its
//! values.
//!
//! The scale of a head: the head's `partials_per_head` partials summed in the
//! kernel's order (one add per partial after the first), the division by the
//! head's columns (exact: a power of two), the epsilon's add (1), and the
//! library's `rsqrt`. The sum's path is `partials_per_head - 1` adds, so the
//! scale's argument is off by at most `t = G(partials_per_head)` relative, and
//! the scale by at most `(1 - t)^(-1/2) - 1` (the worse direction) plus the
//! library `rsqrt`'s own error (`RSQRT_ULP` units in the last place, as in the
//! norm above).
//!
//! The scale multiplies each block's two head groups once at the end of the
//! block (the dot is linear in the input), so its product joins the path (1)
//! and the gated numerator is the input and is exact: the mat-vec's path
//! grows by one operation, `G(n_store + 1)`. The scale's own error is a
//! relative error of every term, so it enters the band the same way the norm's
//! does: the factor is `theta_s + g + theta_s * g` with `g = G(n_store + 1)`.
//! The band is on `sum_c M_c |x_hat_c|`, `x_hat = g * s` in f64, and the
//! residual epilogue adds `U |y|` as in the residual kind.
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

use bloomery_metal::kernels::Quant;
use gguf::quant::dequant_row;

/// Unit roundoff of an f32 operation rounded to nearest.
pub const U: f64 = 1.0 / 16_777_216.0;

/// The most error of the library's `rsqrt`, in units in the last place.
pub const RSQRT_ULP: f64 = 2.0;

/// One value and the absolute error an f32 evaluation may add: the
/// arithmetic of the delta and the short conv referees (`oracle_delta.rs`,
/// `oracle_shortconv.rs`), each operation rounding to nearest once.
#[derive(Clone, Copy)]
pub(crate) struct Ev {
    pub(crate) v: f64,
    pub(crate) e: f64,
}

impl Ev {
    pub(crate) fn zero() -> Self {
        Self { v: 0.0, e: 0.0 }
    }

    pub(crate) fn exact(v: f32) -> Self {
        Self {
            v: f64::from(v),
            e: 0.0,
        }
    }
}

/// A product pays the two linear terms and `ea·eb`.
pub(crate) fn mul(a: Ev, b: Ev) -> Ev {
    let v = a.v * b.v;
    let cross = a.e * b.v.abs() + b.e * a.v.abs() + a.e * b.e;
    let mag = (a.v.abs() + a.e) * (b.v.abs() + b.e);
    Ev {
        v,
        e: cross + U * mag,
    }
}

/// An fma rounds `a·b + c` once.
pub(crate) fn fma(a: Ev, b: Ev, c: Ev) -> Ev {
    let v = a.v.mul_add(b.v, c.v);
    let cross = a.e * b.v.abs() + b.e * a.v.abs() + a.e * b.e;
    let mag = (a.v.abs() + a.e) * (b.v.abs() + b.e) + (c.v.abs() + c.e);
    Ev {
        v,
        e: cross + c.e + U * mag,
    }
}

/// The operations of a Q4_K store path before the lane's blocks and the SIMD
/// group's sum: the `dmin` side's fixed count of the model in this module's
/// doc.
///
/// PIN(2026-10-22): the model in this module's doc, and no measured error,
/// sets the band. (Re-derived the same day, before landing: the scale's
/// `(1 + t)^(-1/2)` is taken exactly, where it was taken to first order with
/// a 1 % margin, and the absolute floor counts its path's operations, where
/// it was a count with no derivation. The norm band narrowed by under half a
/// percent.)
const Q4K_STORE_BASE: usize = 13;

/// The operations of a Q6_K store path before the lane's blocks and the SIMD
/// group's sum: the four-term sum, its scale and its lane count.
const Q6K_STORE_BASE: usize = 9;

/// The operations of an A4 store path before the lane's blocks and the SIMD
/// group's sum: the four-term accumulator chain, the combine, the scale
/// product and the bias add, as this module's doc derives.
const A4_STORE_BASE: usize = 9;

/// The operations of an A8 store path before the lane's blocks and the SIMD
/// group's sum: A4's own count over its own tree, as this module's doc
/// derives (the one-byte codes unpack to their values, so the combine adds
/// carry no power-of-two product).
///
/// PIN(2026-10-30): the model in this module's doc, and no measured error,
/// sets the band.
const A8_STORE_BASE: usize = 9;

/// The operations of a Q8_0 store path before the lane's blocks and the SIMD
/// group's sum: the product, the eight-term chain and the scale product, as
/// this module's doc derives.
///
/// PIN(2026-10-25): the model in this module's doc, and no measured error,
/// sets the band.
const Q8_0_STORE_BASE: usize = 9;

/// The operations of a Q5_K store path before the lane's blocks and the SIMD
/// group's sum: the `d` side's fixed count of the model in this module's doc.
///
/// PIN(2026-10-25): the model in this module's doc, and no measured error,
/// sets the band.
const Q5K_STORE_BASE: usize = 15;

/// The operations of an F32 store path before the lane's blocks and the SIMD
/// group's sum: the product, then the two pair adds of a quarter's four
/// products, as the owner's `partial` orders them.
///
/// PIN(2026-10-27): the model in this module's doc, and no measured error,
/// sets the band.
const F32_STORE_BASE: usize = 3;

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
    /// The head-normed residual mat-vec: the gated delta net's out projection.
    HeadNorm(HeadGeometry),
}

/// The head geometry the head-normed mat-vec's band depends on: the value
/// heads, the columns one head holds, and the partial sums one head's sum is
/// made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGeometry {
    /// The value heads.
    pub heads: usize,
    /// The columns one head holds.
    pub cols_per_head: usize,
    /// The partial sums one head's sum is made of.
    pub partials_per_head: usize,
}

/// What a mat-vec's band is made of (the derivation above): the relative
/// factor of the sum of the terms' magnitudes, and the operations of the
/// longest path, which set the absolute floor.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    factor: f64,
    ops: usize,
}

impl Band {
    /// The relative factor of the sum of the terms' magnitudes.
    #[must_use]
    pub fn factor(self) -> f64 {
        self.factor
    }

    /// The operations of the longest path. The absolute floor is this many
    /// times the smallest positive f32.
    #[must_use]
    pub fn ops(self) -> usize {
        self.ops
    }
}

/// The relative error of `n` rounded operations, `n U / (1 - n U)`.
#[must_use]
pub fn gamma_bound(n: usize) -> f64 {
    gamma(n)
}

/// The band of a mat-vec of `quant` over `nblocks` blocks of columns.
pub fn band(quant: Quant, nblocks: usize, kind: Kind) -> Band {
    let (lane_blocks, base) = match quant {
        Quant::Q4K => (nblocks.div_ceil(4), Q4K_STORE_BASE),
        Quant::Q6K => (nblocks.div_ceil(2), Q6K_STORE_BASE),
        Quant::A4 | Quant::A4H => (nblocks.div_ceil(8), A4_STORE_BASE),
        Quant::A8 | Quant::A8H => (nblocks.div_ceil(8), A8_STORE_BASE),
        Quant::Q8_0 => (nblocks.div_ceil(8), Q8_0_STORE_BASE),
        Quant::Q5K => (nblocks.div_ceil(4), Q5K_STORE_BASE),
        Quant::F32 => (nblocks.div_ceil(8), F32_STORE_BASE),
    };
    let n_store = base + lane_blocks + SIMD_SUM_DEPTH;
    match kind {
        Kind::Store | Kind::Residual => Band {
            factor: gamma(n_store),
            ops: n_store,
        },
        Kind::Norm => {
            let chain = match quant {
                Quant::Q4K | Quant::Q5K => 8 * lane_blocks,
                Quant::Q6K => 4 * lane_blocks,
                Quant::A4 | Quant::A8 | Quant::A4H | Quant::A8H => 4 * lane_blocks,
                Quant::Q8_0 => 2 * lane_blocks,
                Quant::F32 => lane_blocks,
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
        Kind::HeadNorm(heads) => {
            // PIN(2026-10-22): the model in this module's doc sets the
            // head-normed band.
            //
            // The head scale: the head's partial sums summed (one add per
            // partial after the first), the division by the head's columns
            // (exact: a power of two), the epsilon's add, and the rsqrt.
            let n_s = heads.partials_per_head - 1 + 1;
            let t = gamma(n_s);
            let theta_s = (1.0 - t).powf(-0.5) - 1.0 + RSQRT_ULP * 2.0 * U;
            // The scale multiplies each block's two head groups once, so
            // its product joins the path; the numerator is the input.
            let g = gamma(n_store + 1);
            Band {
                factor: theta_s + g + theta_s * g,
                ops: n_s + n_store + 1,
            }
        }
    }
}

impl Band {
    /// The width [`check`] allows one row at `magnitude`, with no residual.
    ///
    /// The same width the norm-on-load and the plain store use: the relative
    /// factor times the terms' magnitudes, and the path's absolute floor.
    #[must_use]
    pub fn row_width(self, magnitude: f64) -> f64 {
        let floor = self.ops as f64 * f64::from(f32::MIN_POSITIVE);
        self.factor * magnitude * (1.0 + U) + floor
    }
}

/// The head scale the head-normed mat-vec applies to a head's columns:
/// `1 / sqrt(sum / cols + eps)`, in f64, from the head's partial sums.
pub fn head_scale(partials: &[f64], cols_per_head: usize, eps: f64) -> f64 {
    let sum: f64 = partials.iter().sum();
    1.0 / (sum / cols_per_head as f64 + eps).sqrt()
}

/// One output row of the oracle: the f64 dot and the sum of the terms'
/// magnitudes the band is a multiple of.
pub struct Reference {
    pub y: Vec<f64>,
    pub magnitude: Vec<f64>,
}

/// `block`'s bytes with `d` made non-negative and `dmin` non-positive: the
/// reader's dequantization of them is `d sc q + |dmin| m`, the magnitude of
/// the two terms of a Q4_K or Q5_K weight (both blocks open with `d` and
/// `dmin`, little-endian f16s).
fn k_magnitude_blocks(quant: Quant, row: &[u8]) -> Vec<u8> {
    let mut out = row.to_vec();
    for block in out.chunks_exact_mut(quant.block_bytes()) {
        block[1] &= 0x7F;
        block[3] |= 0x80;
    }
    out
}

/// `row`'s bytes with the scale's and the bias's sign bits cleared: the
/// reader's dequantization of them is `|s| q + |b|`, the magnitude of the two
/// terms of an affine (A4, A8, A4H, A8H) weight. Each sign bit is the top bit
/// of the pair's second byte (little-endian), at the group's end.
fn affine_magnitude_blocks(quant: Quant, row: &[u8]) -> Vec<u8> {
    let bytes = quant.block_bytes();
    assert!(
        row.len().is_multiple_of(bytes),
        "a {} row of {} bytes is not whole {bytes}-byte groups",
        quant.name(),
        row.len()
    );
    let mut out = row.to_vec();
    for group in out.chunks_exact_mut(bytes) {
        group[bytes - 3] &= 0x7F;
        group[bytes - 1] &= 0x7F;
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
    let row_bytes = cols / quant.values_per_block() * quant.block_bytes();
    if weights.len() != rows * row_bytes || x.len() != cols {
        return Err(format!(
            "oracle shape: {} weight bytes and {} inputs for {rows} x {cols}",
            weights.len(),
            x.len()
        ));
    }
    let ty = quant.ggml();
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
            Quant::Q6K | Quant::Q8_0 | Quant::F32 => w
                .iter()
                .zip(x)
                .map(|(&w, &x)| f64::from(w).abs() * x.abs())
                .sum(),
            Quant::Q4K | Quant::Q5K => {
                dequant_row(ty, &k_magnitude_blocks(quant, row), &mut m)
                    .map_err(|e| format!("dequantizing a row's magnitudes: {e}"))?;
                m.iter().zip(x).map(|(&m, &x)| f64::from(m) * x.abs()).sum()
            }
            Quant::A4 | Quant::A8 | Quant::A4H | Quant::A8H => {
                dequant_row(ty, &affine_magnitude_blocks(quant, row), &mut m)
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
    if matches!(kind, Kind::Residual | Kind::HeadNorm(_)) {
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

/// `1 / (1 + exp(-x))` in f64.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `x / (1 + exp(-x))` in f64.
#[must_use]
pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
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

/// The dequantized values of a quantized table row, the embedding oracle.
pub fn table_row(quant: Quant, table: &[u8], cols: usize, id: usize) -> Result<Vec<f32>, String> {
    let row_bytes = cols / quant.values_per_block() * quant.block_bytes();
    let mut out = vec![0f32; cols];
    dequant_row(
        quant.ggml(),
        &table[id * row_bytes..(id + 1) * row_bytes],
        &mut out,
    )
    .map_err(|e| format!("dequantizing row {id}: {e}"))?;
    Ok(out)
}
