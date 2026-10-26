//! The attention oracle: the kernel's f32 norm and rope, the host
//! quantizers, an f64 softmax over the dequantized cache, and the band, at
//! the host instance's head ([`HEAD_DIM`] values, the first [`ROT`] turning).
//!
//! The norm follows the CUDA append's pairing and warp order: thread `t` of
//! `HEAD/2` owns `owned_pair(HEAD, ROT, t)`, `(t, t + ROT/2)` for
//! `t < ROT/2` and `(t + ROT/2, t + HEAD/2)` otherwise; the xor butterfly is
//! 16, 8, 4, 2, 1 with every lane reading the values from before the step,
//! and the `HEAD/64` warp sums are added in warp order.
//! The CUDA sum is f64. This GPU has no f64, so the kernel and this oracle
//! evaluate that tree in f32. The mean is `sum / HEAD`, the scale is
//! `1 / sqrt(mean + eps)` (two roundings), and a scale that is not finite
//! and positive is NaN. The turn is `fma(n0, c, -(n1 * s))` and
//! `fma(n0, s, n1 * c)` on table pair `i` of the position's row. Values at
//! dim `ROT` and above pass through as `(scale * gain) * x`.
//!
//! A q8_0 row is `gguf::quant::quantize_q8_0` per 32-value block, packed into
//! the kernel's planes (`HEAD/4` code words, low byte first, and `HEAD/32`
//! f16 scales).
//! An f16 row is `gguf::quant::f32_to_f16_bits`.
//!
//! # The band
//!
//! PIN(2026-10-22): the model below, and no measured error, sets the band.
//!
//! A lane owns `HEAD/32` dims and sums them with as many fmas. The SIMD
//! group's sum is at most 31 adds, whichever tree the language picks. The
//! score multiplies by the scale (`1/16` at head 256, exact; the f32
//! `1/sqrt(128)` at head 128, one rounding, and the reference reads the same
//! f32 scale). The longest path from a product `q_i k_i` to the score is
//! `HEAD/32 + 32` roundings (40 at head 256), so a score is off by at most
//! `Δ = G(HEAD/32 + 32) * scale * sum_i |q_i k_i|`, `G(n) = nU / (1 - nU)`,
//! `U = 2^-24`.
//!
//! Metal Shading Language Specification version 4.1, section 8.4, Table 8.1,
//! row `exp`: at most 4 ulp. The build selects that table
//! (`-fmetal-math-mode=safe`, `-fmetal-math-fp32-functions=precise`). One ulp
//! of a normal is at most `2^-23` relative, so 4 ulp is `η = 2^-21` relative.
//! Addition and division in the same table are correctly rounded.
//!
//! Two centered exps each slip by `exp(Δ)`, and each `exp` by `η`:
//! `rel = exp(2Δ) * (1+η) * (1+η) - 1`. The online sum then adds one rounding
//! per key, one rescale per tile and one per segment in the merge:
//! `n_rescale = ceil(n_keys / 16) + segments`, and
//! `θ = G(n_keys + n_rescale) + n_rescale * η`. Dimension `d` of the attention
//! is held to `B = (rel + θ) * sum_j w_j |v_jd|`, `w` the f64 softmax weight.
//!
//! A layer with no gate stores `y = attn`, held to `B` plus the floor. A
//! gated layer stores `y = attn * σ(g)`, `σ(g) = 1 / (1 + exp(-g))`.
//! `|Δσ| ≤ η σ (1-σ) + 2 U σ`: the exp's relative slip through `dσ/de * e`
//! (`e / (1+e) = 1-σ`), then one correctly rounded add and one correctly
//! rounded division. The product adds `U |y|`. The band is
//! `|σ| B + |attn| |Δσ| + U |y|`, plus one `f32::MIN_POSITIVE` per operation
//! on that path (the 40, one per key, one per rescale, the division by the
//! weight sum, and four for the sigmoid and the product). The reference
//! softmax and the reference sigmoid are f64.
//!
//! # The prefill band
//!
//! The prefill kernel rounds the normed and roped query to f16 before the
//! score, so each product `q_i k_i` carries `q_i`'s relative error
//! `2^-11` (round to nearest, normal range) and an absolute `2^-25` for a
//! value in f16's subnormal range. The keys are the cache's f16 words, exact.
//! The `HEAD` products are summed in f32 by the fragment product, in some
//! order of at most `HEAD + 8` roundings with the scale's one: a score is off
//! by at most
//! `Δ = scale * ((2^-11 + G(HEAD + 9)) * sum_i |q_i k_i| + 2^-25 * sum_i |k_i|)`.
//! The weights are then rounded to f16 before the value product, each by a
//! relative `2^-11` and an absolute `2^-25` (the weight sum stays f32), so the
//! attention is held to `B = (rel + 2^-11 + θ) * sum_j w_j |v_jd| + 2^-25 *
//! sum_j |v_jd|` (the weight sum is at least 1 in the kernel's scale, the max
//! weight being 1). The tile rescales are `ceil(n_keys / 16)`, and there are no
//! segments. The gate adds the decode band's terms unchanged.

use bloomery_metal::kernels::{HEAD_DIM, ROT, SCALE, TILE};
use gguf::quant::{f32_to_f16_bits, half_to_f32, quantize_q8_0};

use super::oracle::{U, gamma_bound, sigmoid};
/// Lane fmas, the SIMD-group sum, and the scale multiply.
const DOT_OPS: usize = HEAD_DIM / 32 + 31 + 1;
/// 4 ulp of a normal, as a relative error: `4 * 2^-23`.
const ETA: f64 = 1.0 / 2_097_152.0;
/// The sigmoid and the product: exp, add, divide, multiply.
const GATE_OPS: usize = 4;

fn f64_usize(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("a band count fits u32"))
}

/// The norm's pair tree's warps: `HEAD/2` threads, 32 a warp.
const PAIR_WARPS: usize = HEAD_DIM / 64;

fn warp_sums(x: &[f32; HEAD_DIM]) -> [f32; PAIR_WARPS] {
    let mut acc = [[0f32; 32]; PAIR_WARPS];
    for (w, row) in acc.iter_mut().enumerate() {
        for (lane, slot) in row.iter_mut().enumerate() {
            let tid = w * 32 + lane;
            let (i0, i1) = if tid < ROT / 2 {
                (tid, tid + ROT / 2)
            } else {
                (tid + ROT / 2, tid + HEAD_DIM / 2)
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
    acc.map(|row| row[0])
}

fn rms_scale(sum: f32, eps: f32) -> f32 {
    let mean = sum / f32::from(u16::try_from(HEAD_DIM).expect("a head's width fits u16"));
    let scale = 1.0 / (mean + eps).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        f32::NAN
    } else {
        scale
    }
}

/// Norm then rope of one head, in the kernel's f32 order.
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
    for w in &warps[1..] {
        sum += w;
    }
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

/// The f64 softmax weights of `q` over `keys`, summing to 1.
fn softmax_weights(q: &[f32; HEAD_DIM], keys: &[[f32; HEAD_DIM]]) -> Result<Vec<f64>, String> {
    if keys.is_empty() {
        return Err("attention over no keys".to_string());
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
    let mut weights = Vec::with_capacity(scores.len());
    let mut weight_sum = 0.0f64;
    for s in &scores {
        let w = (s - m).exp();
        weight_sum += w;
        weights.push(w);
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err("the softmax weight sum is empty".to_string());
    }
    for w in &mut weights {
        *w /= weight_sum;
    }
    Ok(weights)
}

/// f64 attention of `q` over `keys` and `vals`: the kernel's scale,
/// softmax, then the weighted values. The result is the attention context,
/// before any gate.
pub fn attend(
    q: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    vals: &[[f32; HEAD_DIM]],
) -> Result<[f64; HEAD_DIM], String> {
    if keys.len() != vals.len() {
        return Err(format!(
            "attention over {} keys and {} values",
            keys.len(),
            vals.len()
        ));
    }
    let weights = softmax_weights(q, keys)?;
    let mut acc = [0.0f64; HEAD_DIM];
    for (j, w) in weights.iter().enumerate() {
        for d in 0..HEAD_DIM {
            acc[d] += w * f64::from(vals[j][d]);
        }
    }
    Ok(acc)
}

/// What the band's derivation takes from the kernel it bounds.
struct Terms {
    /// Relative error of the query operand of a score product.
    operand: f64,
    /// Absolute error of the query operand, per unit of the key's magnitude.
    operand_abs: f64,
    /// Roundings on the longest path from a product to the score.
    dot_ops: usize,
    /// Relative error of a weight before the value product.
    weight: f64,
    /// Absolute error of a weight before the value product.
    weight_abs: f64,
}

/// The decode flash's terms: exact operands and weights.
const DECODE: Terms = Terms {
    operand: 0.0,
    operand_abs: 0.0,
    dot_ops: DOT_OPS,
    weight: 0.0,
    weight_abs: 0.0,
};

/// f16 round to nearest, relative: `2^-11`.
const F16_REL: f64 = 1.0 / 2048.0;

/// f16's subnormal spacing over two: `2^-25`.
const F16_ABS: f64 = 1.0 / 33_554_432.0;

/// The prefill flash's terms: the query and the weights rounded to f16, a
/// fragment product over the head's values (plus the 8 of a fragment's own
/// sum and the scale).
const PREFILL: Terms = Terms {
    operand: F16_REL,
    operand_abs: F16_ABS,
    dot_ops: HEAD_DIM + 8 + 1,
    weight: F16_REL,
    weight_abs: F16_ABS,
};

/// The absolute band per dimension of the output, gated by `gate` when the
/// layer has one, from the derivation above.
pub fn band(
    q: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    vals: &[[f32; HEAD_DIM]],
    gate: Option<&[f32; HEAD_DIM]>,
    segments: usize,
) -> Result<[f64; HEAD_DIM], String> {
    band_of(&DECODE, q, keys, vals, gate, segments)
}

/// The band of the prefill flash, from the prefill derivation above. The
/// prefill has no segments.
pub fn prefill_band(
    q: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    vals: &[[f32; HEAD_DIM]],
    gate: Option<&[f32; HEAD_DIM]>,
) -> Result<[f64; HEAD_DIM], String> {
    band_of(&PREFILL, q, keys, vals, gate, 0)
}

fn band_of(
    t: &Terms,
    q: &[f32; HEAD_DIM],
    keys: &[[f32; HEAD_DIM]],
    vals: &[[f32; HEAD_DIM]],
    gate: Option<&[f32; HEAD_DIM]>,
    segments: usize,
) -> Result<[f64; HEAD_DIM], String> {
    if keys.len() != vals.len() {
        return Err(format!(
            "the band over {} keys and {} values",
            keys.len(),
            vals.len()
        ));
    }
    let weights = softmax_weights(q, keys)?;
    let scale = f64::from(SCALE);
    let g = gamma_bound(t.dot_ops);
    let mut delta = 0.0f64;
    for k in keys {
        let mut mag = 0.0f64;
        let mut key_mag = 0.0f64;
        for i in 0..HEAD_DIM {
            mag += (f64::from(q[i]) * f64::from(k[i])).abs();
            key_mag += f64::from(k[i]).abs();
        }
        let dj = scale * ((t.operand + g) * mag + t.operand_abs * key_mag);
        if dj.total_cmp(&delta).is_gt() {
            delta = dj;
        }
    }
    let rel = (2.0 * delta).exp() * (1.0 + ETA) * (1.0 + ETA) - 1.0;
    let n_keys = keys.len();
    let n_rescale = n_keys.div_ceil(TILE) + segments;
    let theta = gamma_bound(n_keys + n_rescale) + f64_usize(n_rescale) * ETA;
    let gate_ops = if gate.is_some() { GATE_OPS } else { 0 };
    let ops = t.dot_ops + n_keys + n_rescale + 1 + gate_ops;
    let floor = f64::from(f32::MIN_POSITIVE) * f64_usize(ops);
    let mut out = [0.0f64; HEAD_DIM];
    for (d, out) in out.iter_mut().enumerate() {
        let mut weighted = 0.0f64;
        let mut value_mag = 0.0f64;
        let mut attn = 0.0f64;
        for (j, v) in vals.iter().enumerate() {
            let value = f64::from(v[d]);
            weighted += weights[j] * value.abs();
            value_mag += value.abs();
            attn += weights[j] * value;
        }
        let b_attn = (rel + t.weight + theta) * weighted + t.weight_abs * value_mag;
        *out = match gate {
            Some(gate) => {
                let sigma = sigmoid(f64::from(gate[d]));
                let y = attn * sigma;
                let d_sigma = ETA * sigma * (1.0 - sigma) + 2.0 * U * sigma;
                sigma.abs() * b_attn + attn.abs() * d_sigma.abs() + U * y.abs() + floor
            }
            None => b_attn + floor,
        };
    }
    Ok(out)
}
