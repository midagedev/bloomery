//! What the gates' host transcriptions and bands share about float
//! arithmetic: the unit roundoffs, the `γ(n)` bound on a result that went
//! through `n` roundings, and the warp butterfly whose order our kernels sum
//! 32 lanes in.

/// f32's unit roundoff, 2^-24.
pub const U_F32: f32 = f32::EPSILON / 2.0;

/// f32's unit roundoff as an f64, for a bound computed in f64.
pub const U: f64 = U_F32 as f64;

/// f64's unit roundoff, 2^-53.
pub const U64: f64 = f64::EPSILON / 2.0;

/// `γ(n) = n·u / (1 − n·u)` with f32's `u`: the relative bound on a result
/// that went through `n` f32 roundings.
pub fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// The xor butterfly `warp::reduce_sum_f32` runs over 32 lane values (and
/// `warp::shuffle_xor_f64` in f64): for `off` = 16, 8, 4, 2, 1 each lane adds
/// its partner `lane ^ off` to its own value. Lane 0's result.
pub fn butterfly<T: Copy + std::ops::Add<Output = T>>(mut v: [T; 32]) -> T {
    for off in [16, 8, 4, 2, 1] {
        let prev = v;
        for (l, s) in v.iter_mut().enumerate() {
            *s = prev[l] + prev[l ^ off];
        }
    }
    v[0]
}

/// One GEMM epilogue step on the host, bit for bit `bloomery_gpu::gemm`'s
/// `scaled_sum`: `fma(scale, fma(a[1], b[1], a[0]·b[0]), acc)` in F32 while
/// every intermediate stays in the F32 normal range, and otherwise the inner
/// sum (F32 when it stayed in range, else exact) times `scale` plus `acc` as
/// one F64 fused multiply-add (the kernel's `inner * scale + acc` contracts
/// to `fma.rn.f64`), so an intermediate overflow or underflow cannot erase a
/// finite result. A GEMM contract walks its blocks with it (`grouped.rs` module
/// doc, "Range").
#[must_use]
pub fn scaled_sum(a: [f64; 2], b: [f32; 2], scale: f32, acc: f64) -> f64 {
    let first = a[0] as f32 * b[0];
    let inner = (a[1] as f32).mul_add(b[1], first);
    let result = scale.mul_add(inner, acc as f32);
    let inner_in_range = f64::from(a[0] as f32) == a[0]
        && f64::from(a[1] as f32) == a[1]
        && (first.abs() >= f32::MIN_POSITIVE || a[0] == 0.0 || b[0] == 0.0)
        && first.is_finite()
        && (inner.abs() >= f32::MIN_POSITIVE || inner == 0.0)
        && inner.is_finite();
    if inner_in_range
        && f64::from(acc as f32) == acc
        && (result.abs() >= f32::MIN_POSITIVE || (inner == 0.0 && acc == 0.0) || scale == 0.0)
        && result.is_finite()
    {
        f64::from(result)
    } else {
        let inner = if inner_in_range
            && result.is_finite()
            && (result.abs() >= f32::MIN_POSITIVE
                || f64::from(inner) * f64::from(scale) + acc == 0.0)
        {
            f64::from(inner)
        } else {
            a[0] * f64::from(b[0]) + a[1] * f64::from(b[1])
        };
        inner.mul_add(f64::from(scale), acc)
    }
}

/// One head's norm and NeoX turn, in the device rule's sum order.
#[cfg(feature = "gpu")]
#[must_use]
pub fn neox_head<const H: usize, const R: usize>(
    x: &[f32],
    gain: &[f32],
    cs: &[f32],
    eps: f32,
) -> Vec<f32> {
    const { assert!(H >= 64 && H.is_multiple_of(64) && R.is_multiple_of(2) && R <= H) };
    assert_eq!(x.len(), H, "neox_head: input width");
    assert_eq!(gain.len(), H, "neox_head: gain width");
    assert_eq!(cs.len(), R, "neox_head: rope width");
    let warp = |w| {
        butterfly(std::array::from_fn(|l| {
            let (a, b) = bloomery_gpu::rope_neox::owned_pair(H, R, 32 * w + l);
            f64::from(x[a] * x[a]) + f64::from(x[b] * x[b])
        }))
    };
    let sum = (1..H / 64).fold(warp(0), |s, w| s + warp(w));
    let scale = 1.0 / ((sum / H as f64) as f32 + eps).sqrt();
    let mut y: Vec<f32> = x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect();
    for i in 0..R / 2 {
        let (a, b) = (y[i], y[i + R / 2]);
        let (c, s) = (cs[2 * i], cs[2 * i + 1]);
        y[i] = a.mul_add(c, -(b * s));
        y[i + R / 2] = a.mul_add(s, b * c);
    }
    y
}

/// One head's NORM turn in the device rule, with no norm: pair `(x[2i],
/// x[2i + 1])` turned by `(cs[2i], cs[2i + 1])` as `y0 = x0·c − x1·s`, `y1 = x0·s
/// + x1·c`, each product and each sum rounded on its own (host f32 `*`, `+` and
/// `-` are exactly `mul_rn` and `add_rn`, which rustc never contracts).
#[must_use]
pub fn norm_head<const H: usize>(x: &[f32], cs: &[f32]) -> Vec<f32> {
    const { assert!(H >= 2 && H.is_multiple_of(2)) };
    assert_eq!(x.len(), H, "norm_head: input width");
    assert_eq!(cs.len(), H, "norm_head: rope width");
    let mut y = vec![0.0f32; H];
    for i in 0..H / 2 {
        let (x0, x1) = (x[2 * i], x[2 * i + 1]);
        let (c, s) = (cs[2 * i], cs[2 * i + 1]);
        y[2 * i] = x0 * c - x1 * s;
        y[2 * i + 1] = x0 * s + x1 * c;
    }
    y
}

/// A 32-value block's largest magnitude over its RMS, at most: a block of
/// one nonzero value, √32 — the crest [`q8_32_rel`] takes.
pub const CREST_MAX_32: f64 = 5.656_854_249_492_381;

/// The error model of a q8 activation of 32 values (scale `d = amax/127`,
/// codes rounded to nearest): each value moves by at most `d/2`, uniformly,
/// so by `d/√12` in RMS, and the block's RMS error over its RMS is
/// `crest/(127·√12)`, at most [`CREST_MAX_32`]`/(127·√12)` = 1.2858e-2. A
/// projection passes that relative error to its output. Every band a gate
/// derives for an input the 32-value quantizers made starts here.
#[must_use]
pub fn q8_32_rel() -> f64 {
    CREST_MAX_32 / (127.0 * 12f64.sqrt())
}

/// The bound on a `k`-term dot with both operands rounded to BF16 and F32
/// accumulation, as a fraction of its terms' absolute sum `Σ|w·x|`.
pub fn bf16_dot_rel(k: usize) -> f64 {
    let u = 1.0 / 256.0;
    2.0 * u + u * u + gamma(k)
}

/// BF16 operands against exact dequantized rows and the released Q8_0
/// weight/activation representation. KL is over each column's output rows.
pub fn check_mma(tag: &str, (y, x, k): (&[f32], &[f32], usize), w: &[Vec<f32>]) -> bool {
    use crate::prompts::log_softmax;
    use crate::verdict;
    use gguf::quant::quantize_row_q8_0_roundtrip;

    let rows = w.len();
    let mut wq = vec![vec![0.0; k]; rows];
    for (src, dst) in w.iter().zip(&mut wq) {
        quantize_row_q8_0_roundtrip(src, dst);
    }
    let (act, _) = crate::gemm32::host_act(x, k, x.len() / k);
    let xq: Vec<f32> = act
        .codes
        .iter()
        .enumerate()
        .map(|(i, &q)| f32::from(q) * act.d[i / 32])
        .collect();
    let q8_representable = wq.iter().flatten().chain(&xq).all(|v| v.is_finite());
    let mut kl = 0.0f64;
    let mut kl_q8 = 0.0f64;
    let mut worst = 0.0f64;
    let mut band = true;
    let mut err2 = 0.0f64;
    let mut qerr2 = 0.0f64;
    for c in 0..x.len() / k {
        let mut exact = Vec::with_capacity(rows);
        let mut q8 = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut sum = 0.0f64;
            let mut mag = 0.0f64;
            let mut sq = 0.0f64;
            for j in 0..k {
                let term = f64::from(w[r][j]) * f64::from(x[c * k + j]);
                sum += term;
                mag += term.abs();
                sq += f64::from(wq[r][j]) * f64::from(xq[c * k + j]);
            }
            let bound = bf16_dot_rel(k) * mag;
            let err = (f64::from(y[c * rows + r]) - sum).abs();
            band &= err.is_finite() && err <= bound;
            worst = worst.max(if bound > 0.0 { err / bound } else { err });
            err2 += err * err;
            qerr2 += (sq - sum).powi(2);
            exact.push(sum as f32);
            q8.push(sq as f32);
        }
        let p = log_softmax(&exact);
        let q = log_softmax(&y[c * rows..][..rows]);
        let q8 = if q8_representable {
            log_softmax(&q8)
        } else {
            vec![0.0; rows]
        };
        for ((p, q), q8) in p.into_iter().zip(q).zip(q8) {
            kl += p.exp() * (p - q);
            if q8_representable {
                kl_q8 += p.exp() * (p - q8);
            }
        }
    }
    let cols = x.len() / k;
    kl /= cols as f64;
    kl_q8 /= cols as f64;
    // A single dot's L2 error depends on cancellation; KL and the derived band decide acceptance.
    let quality = kl.is_finite()
        && (!q8_representable || (kl_q8.is_finite() && kl.max(0.0) <= kl_q8.max(0.0) + 1e-12));
    let pass = band && quality;
    println!(
        "dense_mma {tag} columns={cols} q8_representable={q8_representable} worst_err_over_band={worst:.6e} \
         kl={kl:.6e} q8_kl={kl_q8:.6e} error_l2={:.6e} q8_error_l2={:.6e} {}",
        err2.sqrt(),
        qerr2.sqrt(),
        verdict(pass)
    );
    pass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mma_rejects_undefined_kl() {
        let mut w = vec![vec![0.0; 64]];
        let mut x = vec![0.0; 64];
        w[0][0] = f32::from_bits(0x7f01_0000);
        x[0] = f32::from_bits(0x3ffe_0419);
        let scalar = check_mma(
            "overflow_scalar",
            (&[f32::from_bits(0x7f7f_fc00)], &x, 64),
            &w,
        );
        let rows = check_mma(
            "overflow_rows",
            (&[f32::MAX, 0.0], &[4.0; 64], 64),
            &[vec![2.0f32.powi(120); 64], vec![0.0; 64]],
        );
        assert!(
            !scalar && !rows,
            "undefined KL accepted: scalar={scalar} rows={rows}"
        );
    }

    /// √32 is the crest's bound, and the model's value is the one the
    /// bands' derivations quote.
    #[test]
    fn the_q8_32_error_model() {
        assert!((CREST_MAX_32 * CREST_MAX_32 - 32.0).abs() < 1e-12);
        assert!((q8_32_rel() - 1.2858e-2).abs() < 1e-6, "{}", q8_32_rel());
    }
}
