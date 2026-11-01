//! The attention oracle: the kernel's f32 norm and rope, the host
//! quantizers, an f64 softmax over the dequantized cache, and the band, at
//! the head its inputs carry: the head's width is the query's length, and the
//! values that turn are the rope row's. An input of another length than the
//! one it pairs with is refused by name.
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
//! PIN(2026-10-29): the one-pass flash (the f16 cache's one segment) sums in
//! another order, and the one-segment band holds it unchanged. Its score is
//! the segmented flash's: the same lane fmas, SIMD sum and scale. Its `n`
//! keys are strided over 16 SIMD groups, so a group adds at most
//! `ceil(n / 16)` weights and rescales once a round of `16 * 4` keys, and the
//! merge adds the live groups' sums by fma, at most `min(n, 16)` of them, each
//! times one factor `exp(m_g - M)`. A product's path is then at most
//! `ceil(n / 16) + ceil(n / 64) - 1 + min(n, 16) <= n + ceil(n / 16) + 1`
//! roundings, and a weight meets at most `ceil(n / 64) <= ceil(n / 16) + 1`
//! exps past its own: inside `G(n_keys + n_rescale)` and `n_rescale * η` at
//! `segments = 1`.
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
//! # The window, the per-head gate and the rope factors
//!
//! A layer's window is the last `window` of its live keys ([`window_tail`]): the
//! softmax runs over those alone, so the band's key and rescale counts are
//! theirs. A per-head gate is the head's one raw value in every channel
//! ([`head_gate_row`]), and the band's gate terms hold unchanged.
//!
//! The kernels read a host table of `(cos, sin)` a pair a position, so the rope
//! factors of ggml's `freq_factors` are the table's recipe ([`rope_row_f64`]):
//! `theta_i = pos * base^(-2i/dims) / f_i`, in f64, one rounding to f32. A
//! factor of `1e30` (f32 `1.0000000150474662e30`) leaves its pair unrotated
//! within f32 and by no special case: `theta_i <= pos` (`theta_0 = pos`), so the
//! angle is at most `pos * 1e-30`, `cos` of it is exactly 1 (`x^2/2` is under
//! `2^-24` times 1 for any `x < 1e-4`), and `sin` of it is `x`, a normal f32
//! (`pos * 1e-30` is above `1.2e-38` for any `pos >= 1`) that is below f32's
//! resolution next to 1 (`1 + x == 1`). The turn `fma(n0, 1, -(n1 * x))` returns
//! `n0` unless `|n0| < |n1| * x * 2^24`, which no normed value of a real layer
//! is.
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

use bloomery_metal::kernels::TILE;
use gguf::quant::{f32_to_f16_bits, half_to_f32, quantize_q8_0};

use super::oracle::{U, gamma_bound, sigmoid};

/// 4 ulp of a normal, as a relative error: `4 * 2^-23`.
const ETA: f64 = 1.0 / 2_097_152.0;
/// The sigmoid and the product: exp, add, divide, multiply.
const GATE_OPS: usize = 4;

fn f64_usize(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("a band count fits u32"))
}

/// The score's scale for a head of `head` values: ik's `1.0f/sqrtf(float(head))`
/// in f32.
fn scale_of(head: usize) -> f32 {
    1.0 / (head as f32).sqrt()
}

/// The head's values fill the norm's pair tree: `head / 2` threads, 32 a
/// warp.
fn check_head(head: usize) -> Result<(), String> {
    if head == 0 || !head.is_multiple_of(64) {
        return Err(format!(
            "a head of {head} values does not fill the pair tree's 32-lane warps"
        ));
    }
    Ok(())
}

/// `x` and `other`, two inputs of the same head, are of one width.
fn same_head(what: &str, head: usize, other: usize) -> Result<(), String> {
    if other != head {
        return Err(format!(
            "{what} holds {other} values, the head holds {head}"
        ));
    }
    Ok(())
}

/// The norm's pair tree's warps, one sum each: `head / 2` threads, 32 a
/// warp, a thread owning the pair `(t, t + rot/2)` while the rope turns it and
/// `(t + rot/2, t + head/2)` after.
fn warp_sums(x: &[f32], rot: usize) -> Vec<f32> {
    let head = x.len();
    let mut acc = vec![[0f32; 32]; head / 64];
    for (w, row) in acc.iter_mut().enumerate() {
        for (lane, slot) in row.iter_mut().enumerate() {
            let tid = w * 32 + lane;
            let (i0, i1) = if tid < rot / 2 {
                (tid, tid + rot / 2)
            } else {
                (tid + rot / 2, tid + head / 2)
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
    acc.iter().map(|row| row[0]).collect()
}

fn rms_scale(sum: f32, eps: f32, head: usize) -> f32 {
    let mean = sum / f32::from(u16::try_from(head).expect("a head's width fits u16"));
    let scale = 1.0 / (mean + eps).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        f32::NAN
    } else {
        scale
    }
}

/// Norm then rope of one head, in the kernel's f32 order. The head is `x`'s
/// width; the rope turns the first `table_row.len()` values.
pub fn norm_rope(x: &[f32], gain: &[f32], eps: f32, table_row: &[f32]) -> Result<Vec<f32>, String> {
    let head = x.len();
    let rot = table_row.len();
    check_head(head)?;
    same_head("a gain", head, gain.len())?;
    if rot == 0 || !rot.is_multiple_of(2) || rot > head {
        return Err(format!(
            "a rope row holds {rot} values, which do not turn pairs of a head of {head}"
        ));
    }
    let warps = warp_sums(x, rot);
    let mut sum = warps[0];
    for w in &warps[1..] {
        sum += w;
    }
    let scale = rms_scale(sum, eps, head);
    let n: Vec<f32> = (0..head)
        .map(|i| {
            let scaled_gain = scale * gain[i];
            scaled_gain * x[i]
        })
        .collect();
    let mut y = n.clone();
    for i in 0..rot / 2 {
        let n0 = n[i];
        let n1 = n[i + rot / 2];
        let c = table_row[2 * i];
        let s = table_row[2 * i + 1];
        y[i] = n0.mul_add(c, -(n1 * s));
        y[i + rot / 2] = n0.mul_add(s, n1 * c);
    }
    Ok(y)
}

/// Rope of one head with no head norm, in the kernel's f32 order: the raw
/// values turned as neighbours, pair `i` the values `2i, 2i + 1` (GGUF's NORM
/// pairing) by the table row's pair `i`, the values past the turned width
/// passed through. No sum of squares and no gain: a head of the no-norm
/// prologue is exact in its inputs' width.
pub fn rope_norm_pairs(x: &[f32], table_row: &[f32]) -> Result<Vec<f32>, String> {
    let head = x.len();
    let rot = table_row.len();
    check_head(head)?;
    if rot == 0 || !rot.is_multiple_of(2) || rot > head {
        return Err(format!(
            "a rope row holds {rot} values, which do not turn pairs of a head of {head}"
        ));
    }
    let mut y = x.to_vec();
    for i in 0..rot / 2 {
        let n0 = x[2 * i];
        let n1 = x[2 * i + 1];
        let c = table_row[2 * i];
        let s = table_row[2 * i + 1];
        y[2 * i] = n0.mul_add(c, -(n1 * s));
        y[2 * i + 1] = n0.mul_add(s, n1 * c);
    }
    Ok(y)
}

/// One head as f16 bits, `gguf::quant::f32_to_f16_bits` per value.
#[must_use]
pub fn pack_f16(row: &[f32]) -> Vec<u16> {
    row.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

/// One head as q8_0 planes, `gguf::quant::quantize_q8_0` per block of 32.
///
/// The code word `b * 8 + j / 4` holds code `j` in byte `j % 4`, low byte
/// first. The scale word `b` is the block's `d`.
///
/// # Panics
///
/// When the head is not whole 32-value blocks: no cache row is.
#[must_use]
pub fn pack_q8(row: &[f32]) -> (Vec<u32>, Vec<u16>) {
    assert!(
        !row.is_empty() && row.len().is_multiple_of(32),
        "a q8_0 row is whole 32-value blocks, not {} values",
        row.len()
    );
    let blocks = row.len() / 32;
    let mut codes = vec![0u32; blocks * 8];
    let mut scales = vec![0u16; blocks];
    for b in 0..blocks {
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
///
/// # Panics
///
/// When the scales are not one a block of the codes' 32 values.
#[must_use]
pub fn dequant_q8(codes: &[u32], scales: &[u16]) -> Vec<f32> {
    let head = codes.len() * 4;
    assert_eq!(
        scales.len() * 32,
        head,
        "a q8_0 row holds a scale for each block of 32 values"
    );
    (0..head)
        .map(|dim| {
            let word = codes[dim / 4];
            let shift = (dim % 4) * 8;
            let byte = u8::try_from((word >> shift) & 0xff).expect("a masked byte fits u8");
            let code = i8::from_le_bytes([byte]);
            f32::from(code) * half_to_f32(scales[dim / 32])
        })
        .collect()
}

/// The f64 softmax weights of `q` over `keys`, summing to 1.
fn softmax_weights<R: AsRef<[f32]>>(q: &[f32], keys: &[R]) -> Result<Vec<f64>, String> {
    if keys.is_empty() {
        return Err("attention over no keys".to_string());
    }
    let scale = f64::from(scale_of(q.len()));
    let mut scores = Vec::with_capacity(keys.len());
    let mut m = f64::NEG_INFINITY;
    for k in keys {
        let k = k.as_ref();
        same_head("a key", q.len(), k.len())?;
        let mut dot = 0.0f64;
        for (&a, &b) in q.iter().zip(k) {
            dot += f64::from(a) * f64::from(b);
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
pub fn attend<R: AsRef<[f32]>>(q: &[f32], keys: &[R], vals: &[R]) -> Result<Vec<f64>, String> {
    if keys.len() != vals.len() {
        return Err(format!(
            "attention over {} keys and {} values",
            keys.len(),
            vals.len()
        ));
    }
    let weights = softmax_weights(q, keys)?;
    let mut acc = vec![0.0f64; q.len()];
    for (w, v) in weights.iter().zip(vals) {
        let v = v.as_ref();
        same_head("a value", q.len(), v.len())?;
        for (acc, &value) in acc.iter_mut().zip(v) {
            *acc += w * f64::from(value);
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

/// f16 round to nearest, relative: `2^-11`.
const F16_REL: f64 = 1.0 / 2048.0;

/// f16's subnormal spacing over two: `2^-25`.
const F16_ABS: f64 = 1.0 / 33_554_432.0;

/// The decode flash's terms for a head of `head` values: exact operands and
/// weights, a lane's `head / 32` fmas, the SIMD group's sum, and the scale
/// multiply.
fn decode_terms(head: usize) -> Terms {
    Terms {
        operand: 0.0,
        operand_abs: 0.0,
        dot_ops: head / 32 + 31 + 1,
        weight: 0.0,
        weight_abs: 0.0,
    }
}

/// The prefill flash's terms for a head of `head` values: the query and the
/// weights rounded to f16, a fragment product over the head's values (plus
/// the 8 of a fragment's own sum and the scale).
fn prefill_terms(head: usize) -> Terms {
    Terms {
        operand: F16_REL,
        operand_abs: F16_ABS,
        dot_ops: head + 8 + 1,
        weight: F16_REL,
        weight_abs: F16_ABS,
    }
}

/// The absolute band per dimension of the output, gated by `gate` when the
/// layer has one, from the derivation above.
pub fn band<R: AsRef<[f32]>, G: AsRef<[f32]> + ?Sized>(
    q: &[f32],
    keys: &[R],
    vals: &[R],
    gate: Option<&G>,
    segments: usize,
) -> Result<Vec<f64>, String> {
    band_of(&decode_terms(q.len()), q, keys, vals, gate, segments)
}

/// The band of the prefill flash, from the prefill derivation above. The
/// prefill has no segments.
pub fn prefill_band<R: AsRef<[f32]>, G: AsRef<[f32]> + ?Sized>(
    q: &[f32],
    keys: &[R],
    vals: &[R],
    gate: Option<&G>,
) -> Result<Vec<f64>, String> {
    band_of(&prefill_terms(q.len()), q, keys, vals, gate, 0)
}

fn band_of<R: AsRef<[f32]>, G: AsRef<[f32]> + ?Sized>(
    t: &Terms,
    q: &[f32],
    keys: &[R],
    vals: &[R],
    gate: Option<&G>,
    segments: usize,
) -> Result<Vec<f64>, String> {
    if keys.len() != vals.len() {
        return Err(format!(
            "the band over {} keys and {} values",
            keys.len(),
            vals.len()
        ));
    }
    let head = q.len();
    let gate: Option<&[f32]> = gate.map(|g| g.as_ref());
    if let Some(gate) = gate {
        same_head("a gate", head, gate.len())?;
    }
    let weights = softmax_weights(q, keys)?;
    let scale = f64::from(scale_of(head));
    let g = gamma_bound(t.dot_ops);
    let mut delta = 0.0f64;
    for k in keys {
        let k = k.as_ref();
        let mut mag = 0.0f64;
        let mut key_mag = 0.0f64;
        for (&a, &b) in q.iter().zip(k) {
            mag += (f64::from(a) * f64::from(b)).abs();
            key_mag += f64::from(b).abs();
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
    let mut out = vec![0.0f64; head];
    for (d, out) in out.iter_mut().enumerate() {
        let mut weighted = 0.0f64;
        let mut value_mag = 0.0f64;
        let mut attn = 0.0f64;
        for (j, v) in vals.iter().enumerate() {
            let v = v.as_ref();
            same_head("a value", head, v.len())?;
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

/// The keys a layer attends: the last `window` of `rows`, all of them with no
/// window or one of at least their count.
#[must_use]
pub fn window_tail<R>(rows: &[R], window: Option<u32>) -> &[R] {
    match window {
        Some(w) => {
            let w = usize::try_from(w).expect("a window fits usize");
            &rows[rows.len().saturating_sub(w)..]
        }
        None => rows,
    }
}

/// A per-head gate as the per-channel values [`band`] takes: the head's one raw
/// value in each of its `head` channels.
#[must_use]
pub fn head_gate_row(g: f32, head: usize) -> Vec<f32> {
    vec![g; head]
}

/// The rope row of position `pos` for the NeoX pairs `(i, i + rot / 2)`: `rot`
/// values, a `(cos, sin)` per pair, from ggml's recipe with `freq_factors`:
/// `theta_i = pos * base^(-2i/dims) / f_i`, in f64 and rounded once to f32.
/// `dims` is the width the exponent runs over (the head's, when the whole head
/// turns), `rot` the width the row holds, and `factors` one divisor a pair
/// (`rot / 2`), none for the plain rope.
///
/// # Errors
///
/// A `rot` or `dims` that is zero or a `rot` that is not whole pairs, or
/// factors of another length than the pairs or any that is not finite and
/// positive.
pub fn rope_row_f64(
    pos: usize,
    base: f64,
    dims: usize,
    rot: usize,
    factors: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    if rot == 0 || dims == 0 || !rot.is_multiple_of(2) {
        return Err(format!(
            "a rope row of {rot} values over {dims} dims is not whole pairs of a width"
        ));
    }
    let pairs = rot / 2;
    if let Some(f) = factors {
        if f.len() != pairs {
            return Err(format!(
                "{} rope factors for {pairs} pairs: one a pair",
                f.len()
            ));
        }
        if let Some(bad) = f.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(format!("a rope factor {bad} is not finite and positive"));
        }
    }
    let pos = f64_usize(pos);
    let mut row = Vec::with_capacity(rot);
    for i in 0..pairs {
        let exp = -2.0 * f64_usize(i) / f64_usize(dims);
        let mut theta = pos * base.powf(exp);
        if let Some(f) = factors {
            theta /= f64::from(f[i]);
        }
        row.push(theta.cos() as f32);
        row.push(theta.sin() as f32);
    }
    Ok(row)
}
