//! The attention oracle: the kernel's f32 norm and partial rope, the host
//! quantizers, an f64 softmax over the dequantized cache, and the band.
//!
//! The norm follows the CUDA partial append's pairing and warp order. Pairs
//! are `(i, i + 32)` for `i < 32` and `(tid + 32, tid + 128)` otherwise, the
//! xor butterfly is 16, 8, 4, 2, 1 with every lane reading the values from
//! before the step, and the four warp sums are added in order `0 + 1 + 2 + 3`.
//! The CUDA sum is f64. This GPU has no f64, so the kernel and this oracle
//! evaluate that tree in f32. The mean is `sum / 256`, the scale is
//! `1 / sqrt(mean + eps)` (two roundings), and a scale that is not finite
//! and positive is NaN. The turn is `fma(n0, c, -(n1 * s))` and
//! `fma(n0, s, n1 * c)` on table pair `i` of the position's row. Values at
//! dim 64 and above pass through as `(scale * gain) * x`.
//!
//! A q8_0 row is `gguf::quant::quantize_q8_0` per 32-value block, packed into
//! the kernel's planes (64 code words, low byte first, and 8 f16 scales).
//! An f16 row is `gguf::quant::f32_to_f16_bits`.
//!
//! # The band
//!
//! PIN(2026-10-22): the model below, and no measured error, sets the band.
//!
//! A lane owns 8 dims and sums them with 8 fmas. The SIMD group's sum is at
//! most 31 adds, whichever tree the language picks. The score multiplies by
//! the exact scale `1/16`. The longest path from a product `q_i k_i` to the
//! score is 40 roundings, so a score is off by at most
//! `G(40) * scale * sum_i |q_i k_i|`, `G(n) = nU / (1 - nU)`, `U = 2^-24`.
//! The library `exp` is taken as 4 ulp of a magnitude-1 f32
//! (`4 * 2^-23 = 2^-21`) added to that score error. This exp premise is the
//! first term to question when a clause misses: no source for Metal's
//! precise `exp` bound was read.
//!
//! Logits perturbed by at most `Δ` move the softmax weights by at most
//! `exp(2Δ) - 1` relative (each weight and the partition can each slip by
//! `exp(Δ)`). The online sum then adds one rounding per key, one rescale
//! per tile and one per segment in the merge:
//! `n_rescale = ceil(n_keys / 16) + segments`, and
//! `θ = G(n_keys + n_rescale) + n_rescale * η` with `η` the exp term.
//! Dimension `d` is held to
//! `(exp(2Δ) - 1 + θ) * sum_j |v_jd|`, plus the absolute floor of one
//! `f32::MIN_POSITIVE` per operation on that path. The reference softmax is
//! f64. S = 1 and S > 1 are two evaluations of this band; the triangle
//! through the reference is the sum of the two bounds.

use bloomery_metal::kernels::{HEAD_DIM, ROT, SCALE, TILE};
use gguf::quant::{f32_to_f16_bits, half_to_f32, quantize_q8_0};

const U: f64 = 1.0 / 16_777_216.0;
/// Lane fmas, the SIMD-group sum, and the scale multiply.
const DOT_OPS: usize = 8 + 31 + 1;
/// 4 ulp of a magnitude-1 f32, as an absolute score error.
const ETA: f64 = 1.0 / 2_097_152.0;

fn f64_usize(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("a band count fits u32"))
}

fn gamma(n: usize) -> f64 {
    let nu = f64_usize(n) * U;
    nu / (1.0 - nu)
}

fn warp_sums(x: &[f32; HEAD_DIM]) -> [f32; 4] {
    let mut acc = [[0f32; 32]; 4];
    for (w, row) in acc.iter_mut().enumerate() {
        for (lane, slot) in row.iter_mut().enumerate() {
            let tid = w * 32 + lane;
            let (i0, i1) = if tid < 32 {
                (tid, tid + 32)
            } else {
                (tid + 32, tid + 128)
            };
            *slot = x[i0] * x[i0] + x[i1] * x[i1];
        }
    }
    for mask in [16usize, 8, 4, 2, 1] {
        for row in &mut acc {
            let old = *row;
            for (lane, slot) in row.iter_mut().enumerate() {
                *slot = old[lane] + old[lane ^ mask];
            }
        }
    }
    [acc[0][0], acc[1][0], acc[2][0], acc[3][0]]
}

fn rms_scale(sum: f32, eps: f32) -> f32 {
    let mean = sum / f32::from(u16::try_from(HEAD_DIM).expect("256 fits"));
    let scale = 1.0 / (mean + eps).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        f32::NAN
    } else {
        scale
    }
}

/// Norm then partial rope of one head, in the kernel's f32 order.
pub fn norm_rope(
    x: &[f32; HEAD_DIM],
    gain: &[f32; HEAD_DIM],
    eps: f32,
    table_row: &[f32],
) -> Result<[f32; HEAD_DIM], String> {
    if table_row.len() != ROT {
        return Err(format!(
            "a rope row holds {} values, the partial holds {ROT}",
            table_row.len()
        ));
    }
    let warps = warp_sums(x);
    let mut sum = warps[0];
    sum += warps[1];
    sum += warps[2];
    sum += warps[3];
    let scale = rms_scale(sum, eps);
    let mut n = [0f32; HEAD_DIM];
    for i in 0..HEAD_DIM {
        let scaled_gain = scale * gain[i];
        n[i] = scaled_gain * x[i];
    }
    let mut y = n;
    for i in 0..ROT / 2 {
        let n0 = n[i];
        let n1 = n[i + ROT / 2];
        let c = table_row[2 * i];
        let s = table_row[2 * i + 1];
        y[i] = n0.mul_add(c, -(n1 * s));
        y[i + ROT / 2] = n0.mul_add(s, n1 * c);
    }
    Ok(y)
}

/// One head as f16 bits, `gguf::quant::f32_to_f16_bits` per value.
#[must_use]
pub fn pack_f16(row: &[f32; HEAD_DIM]) -> [u16; HEAD_DIM] {
    let mut out = [0u16; HEAD_DIM];
    for (out, &v) in out.iter_mut().zip(row.iter()) {
        *out = f32_to_f16_bits(v);
    }
    out
}

/// One head as q8_0 planes, `gguf::quant::quantize_q8_0` per block of 32.
///
/// The code word `b * 8 + j / 4` holds code `j` in byte `j % 4`, low byte
/// first. The scale word `b` is the block's `d`.
pub fn pack_q8(row: &[f32; HEAD_DIM]) -> ([u32; HEAD_DIM / 4], [u16; HEAD_DIM / 32]) {
    let mut codes = [0u32; HEAD_DIM / 4];
    let mut scales = [0u16; HEAD_DIM / 32];
    for b in 0..HEAD_DIM / 32 {
        let block = quantize_q8_0(&row[b * 32..b * 32 + 32]);
        scales[b] = block.d;
        for (j, code) in block.q.iter().enumerate() {
            let word = b * 8 + j / 4;
            let shift = (j % 4) * 8;
            codes[word] |= u32::from(code.to_le_bytes()[0]) << shift;
        }
    }
    (codes, scales)
}

/// One f16 value widened, `gguf::quant::half_to_f32`.
#[must_use]
pub fn dequant_f16(bits: u16) -> f32 {
    half_to_f32(bits)
}

/// One q8_0 head dequantized: sign-extended code times `half_to_f32` of its scale.
#[must_use]
pub fn dequant_q8(codes: &[u32], scales: &[u16]) -> [f32; HEAD_DIM] {
    let mut out = [0f32; HEAD_DIM];
    for (dim, out) in out.iter_mut().enumerate() {
        let word = codes[dim / 4];
        let shift = (dim % 4) * 8;
        let byte = u8::try_from((word >> shift) & 0xff).expect("a masked byte fits u8");
        let code = i8::from_le_bytes([byte]);
        *out = f32::from(code) * half_to_f32(scales[dim / 32]);
    }
    out
}

/// f64 attention of `q` over `keys` and `vals`: scale `1/16`, softmax, then
/// the weighted values. The result is the attention context.
pub fn attend(
    q: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    vals: &[[f32; HEAD_DIM]],
) -> Result<[f64; HEAD_DIM], String> {
    if keys.len() != vals.len() || keys.is_empty() {
        return Err(format!(
            "attention over {} keys and {} values",
            keys.len(),
            vals.len()
        ));
    }
    let scale = f64::from(SCALE);
    let mut scores = Vec::with_capacity(keys.len());
    let mut m = f64::NEG_INFINITY;
    for k in keys {
        let mut dot = 0.0f64;
        for i in 0..HEAD_DIM {
            dot += f64::from(q[i]) * f64::from(k[i]);
        }
        let s = dot * scale;
        if s.total_cmp(&m).is_gt() {
            m = s;
        }
        scores.push(s);
    }
    let mut acc = [0.0f64; HEAD_DIM];
    let mut weight_sum = 0.0f64;
    for (j, s) in scores.iter().enumerate() {
        let w = (s - m).exp();
        weight_sum += w;
        for d in 0..HEAD_DIM {
            acc[d] += w * f64::from(vals[j][d]);
        }
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err("the softmax weight sum is empty".to_string());
    }
    Ok(acc.map(|a| a / weight_sum))
}

/// The absolute band per dimension, from the derivation above.
#[must_use]
pub fn band(
    q: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    vals: &[[f32; HEAD_DIM]],
    segments: usize,
) -> [f64; HEAD_DIM] {
    let scale = f64::from(SCALE);
    let g = gamma(DOT_OPS);
    let mut delta = 0.0f64;
    for k in keys {
        let mut mag = 0.0f64;
        for i in 0..HEAD_DIM {
            mag += (f64::from(q[i]) * f64::from(k[i])).abs();
        }
        let dj = g * scale * mag;
        if dj.total_cmp(&delta).is_gt() {
            delta = dj;
        }
    }
    let err = delta + ETA;
    let rel = (2.0 * err).exp() - 1.0;
    let n_keys = keys.len();
    let n_rescale = n_keys.div_ceil(TILE) + segments;
    let theta = gamma(n_keys + n_rescale) + f64_usize(n_rescale) * ETA;
    let ops = DOT_OPS + n_keys + n_rescale + 1;
    let floor = f64::from(f32::MIN_POSITIVE) * f64_usize(ops);
    let mut out = [0.0f64; HEAD_DIM];
    for (d, out) in out.iter_mut().enumerate() {
        let mut mag = 0.0f64;
        for v in vals {
            mag += f64::from(v[d]).abs();
        }
        *out = (rel + theta) * mag + floor;
    }
    out
}
