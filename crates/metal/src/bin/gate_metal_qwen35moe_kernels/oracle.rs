//! The gate's oracle: the fixture's real tensors, `crates/gguf`'s dequantization
//! of the same blocks, an f64 dot on the host, and the band an f32 evaluation
//! of that dot is held to.
//!
//! # The band
//!
//! PIN(2026-10-22): the model below, and no measured error, sets the band.
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
//! by `(1 + t)^(-1/2)` for `t = G(nS + 2)`, and by the library `rsqrt`'s own
//! error, taken as `RSQRT_ULP` units in the last place (2 ulp = 4U). The
//! `RSQRT_ULP` premise is `[unverified]`: no source for Metal's `rsqrt` bound
//! was read, so it is the first term to question when the norm clause misses.
//! The band is on `sum_c M_c |x_hat_c|`, `x_hat = x * gain * s` in f64.
//!
//! **Residual** adds the final `y + dot`: one rounding of the sum, `U |y|`.
//!
//! The oracle's own sums are f64 (`2^-53`), eight orders below `U`, and its
//! dequantized weights are the reader's f32 values, exact up to the one
//! rounding the reader itself makes.

use std::path::{Path, PathBuf};

use bloomery_metal::kernels::{BLOCK_VALUES, Quant};
use gguf::Gguf;
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

/// The relative factor of the band for a mat-vec of `quant` over `nblocks`
/// blocks of columns (the derivation above).
pub fn band_factor(quant: Quant, nblocks: usize, kind: Kind) -> f64 {
    let (lane_blocks, base) = match quant {
        Quant::Q4K => (nblocks.div_ceil(4), 13),
        Quant::Q6K => (nblocks.div_ceil(2), 9),
    };
    let n_store = base + lane_blocks + SIMD_SUM_DEPTH;
    match kind {
        Kind::Store | Kind::Residual => gamma(n_store),
        Kind::Norm => {
            let chain = match quant {
                Quant::Q4K => 8 * lane_blocks,
                Quant::Q6K => 4 * lane_blocks,
            };
            let n_s = 1 + 2 + chain + SIMD_SUM_DEPTH;
            let theta_s = 0.5 * 1.01 * gamma(n_s + 2) + RSQRT_ULP * 2.0 * U;
            // The gain product joins the path, and the scale product ends it.
            let g = gamma(n_store + 2);
            theta_s + g + theta_s * g
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
    factor: f64,
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
    let floor = (SIMD_SUM_DEPTH + 64) as f64 * f64::from(f32::MIN_POSITIVE);
    let mut worst = 0f64;
    for (r, &got) in got.iter().enumerate() {
        let carried = base.map_or(0.0, |b| f64::from(b[r]));
        let want = carried + reference.y[r];
        let band = factor * reference.magnitude[r] * (1.0 + U) + residual_term(kind, want) + floor;
        let err = (f64::from(got) - want).abs();
        if !got.is_finite() || err > band {
            return Err(format!(
                "row {r}: got {got:e}, want {want:e}, error {err:e} over the band {band:e}"
            ));
        }
        worst = worst.max(err / band);
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

/// The fixture file: the qwen35moe trunk the gate reads its tensors from.
pub struct Fixture {
    file: Gguf,
    pub path: PathBuf,
}

impl Fixture {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = Gguf::open(path).map_err(|e| {
            format!(
                "the fixture {} did not open: {e} (copy the qwen35moe fixture's trunk file there, or name it with --fixture)",
                path.display()
            )
        })?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// The bytes of the 2-D tensor `name`, `rows` rows of `cols` columns of
    /// type `ty`, or the first rows of it.
    pub fn rows(
        &self,
        name: &str,
        ty: GgmlType,
        cols: usize,
        rows: usize,
    ) -> Result<&[u8], String> {
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != ty || t.dims.len() < 2 || t.dims[0] != cols as u64 || t.dims[1] < rows as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {ty:?} with {cols} columns and at least {rows} rows",
                t.ty, t.dims
            ));
        }
        let all = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        let row_bytes = cols / BLOCK_VALUES * quant_block_bytes(ty)?;
        Ok(&all[..rows * row_bytes])
    }

    /// The bytes of rows `at..at + rows` of the tensor `name` (see [`Fixture::rows`]).
    pub fn rows_at(
        &self,
        name: &str,
        ty: GgmlType,
        cols: usize,
        at: usize,
        rows: usize,
    ) -> Result<&[u8], String> {
        let all = self.rows(name, ty, cols, at + rows)?;
        let row_bytes = cols / BLOCK_VALUES * quant_block_bytes(ty)?;
        Ok(&all[at * row_bytes..])
    }

    /// The f32 tensor `name` of `len` values.
    pub fn f32s(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != GgmlType::F32 || t.dims != [len as u64] {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {len} f32s",
                t.ty, t.dims
            ));
        }
        let bytes = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        let mut out = vec![0f32; len];
        dequant_row(GgmlType::F32, bytes, &mut out).map_err(|e| format!("{name}: {e}"))?;
        Ok(out)
    }

    /// The model's norm epsilon.
    pub fn rms_eps(&self) -> Result<f32, String> {
        self.file
            .arch_get_f32("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| "the fixture names no attention.layer_norm_rms_epsilon".to_string())
    }

    /// The bytes of the Q4_K embedding table: `rows` rows of `cols`.
    pub fn table(&self, name: &str, cols: usize) -> Result<(&[u8], usize), String> {
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != GgmlType::Q4_K || t.dims.len() != 2 || t.dims[0] != cols as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads a Q4_K table of {cols} columns",
                t.ty, t.dims
            ));
        }
        let bytes = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        Ok((bytes, t.dims[1] as usize))
    }
}

fn quant_block_bytes(ty: GgmlType) -> Result<usize, String> {
    match ty {
        GgmlType::Q4_K => Ok(Quant::Q4K.block_bytes()),
        GgmlType::Q6_K => Ok(Quant::Q6K.block_bytes()),
        other => Err(format!("{other:?} is not a mat-vec type")),
    }
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
