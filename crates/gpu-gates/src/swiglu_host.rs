//! The CPU tier's twin of a gate·up rule (`bloomery_gpu::kquant::Act`), for the
//! gates that hold the card's rows to it: ik's clamped SwiGLU through
//! `qdot::swiglu_clamp` and the OAI SwiGLU through `qdot::swiglu_oai`, each with
//! the rows its limit bit on, and the OAI rule's f64 formula and its band. The
//! plain `silu_mul` has no twin here: the card's elementwise `swiglu` is each
//! gate's own reference for it.

use bloomery_gpu::kquant::Act;
use bloomery_gpu::kquant::act::{SWIGLU_OAI_ALPHA, silu_ik};

/// The relative band of [`Act::SwigluOai`] on the card (or on the CPU tier)
/// against [`oai_exact`]: the exponential's 2 ulp (`expf_ik` sweeps to 1.84), and
/// the sum `1 + e`, the quotient and the product half an ulp each — 3.5 · 2^-23
/// of the result, past the overflow cliff [`oai_flush`] and the smallest normal.
/// The measured worst over 4·10^7 pairs short of the cliff is 3.06 · 2^-23.
pub const OAI_REL: f64 = 3.5 / 8_388_608.0;

/// What [`Act::SwigluOai`] flushes: past `ln(f32::MAX)` = 88.7228 the f32
/// exponential of `alpha · (−x)` is +inf, on the card and in libm alike, and the
/// quotient is −0 where the exact value is `x · (y + 1) / e^z`, largest at the
/// cliff: `x = 88.73 / alpha`, `y + 1 <= limit + 1`, `e^z = f32::MAX`.
#[must_use]
pub fn oai_flush(alpha: f32, limit: f32) -> f64 {
    88.73 / f64::from(alpha) * (f64::from(limit) + 1.0) / f64::from(f32::MAX)
}

/// [`Act::SwigluOai`] at the references' `alpha`.
#[must_use]
pub fn oai(limit: f32) -> Act {
    Act::SwigluOai {
        alpha: SWIGLU_OAI_ALPHA,
        limit,
    }
}

/// `rule` on the sums `g` and `u` as the CPU tier computes it, and the rows its
/// limit bit on: the clamp where `silu(g)` or `|u|` passes the limit, the OAI
/// form where `g` or `|u|` does. `None` for [`Act::SiluMul`].
///
/// # Panics
///
/// On a sum that is not finite, which the clamps would hide.
#[must_use]
pub fn combine(rule: Act, g: &[f32], u: &[f32]) -> Option<(Vec<f32>, usize)> {
    let mut out = vec![0.0f32; g.len()];
    let bit = |over: &dyn Fn(f32) -> bool, limit: f32| {
        g.iter()
            .zip(u)
            .filter(|&(&g, &u)| over(g) || u.abs() > limit)
            .count()
    };
    match rule {
        Act::SiluMul => None,
        Act::SwigluClamp { limit } => {
            assert!(
                qdot::swiglu_clamp(g, u, limit, &mut out),
                "swiglu_clamp: a sum that is not finite"
            );
            Some((out, bit(&|g| silu_ik(g) > limit, limit)))
        }
        Act::SwigluOai { alpha, limit } => {
            assert!(
                qdot::swiglu_oai(g, u, alpha, limit, &mut out),
                "swiglu_oai: a sum that is not finite"
            );
            Some((out, bit(&|g| g > limit, limit)))
        }
    }
}

/// The OAI SwiGLU of one pair in f64: `x / (1 + e^z) · (y + 1)` with `x =
/// min(g, limit)`, `y = clamp(u, ±limit)` and `z = alpha · (−x)` rounded to f32,
/// the argument the reference's `expf` takes.
#[must_use]
pub fn oai_exact(g: f32, u: f32, alpha: f32, limit: f32) -> f64 {
    let x = g.min(limit);
    let z = f64::from(alpha * -x);
    let y = f64::from(u).clamp(-f64::from(limit), f64::from(limit));
    f64::from(x) / (1.0 + z.exp()) * (y + 1.0)
}

/// How many rows of `got` are farther from [`oai_exact`] than [`OAI_REL`] of it
/// (and [`oai_flush`] and the smallest normal, for the values that flush or
/// underflow), and the worst distance in bands.
#[must_use]
pub fn oai_misses(got: &[f32], g: &[f32], u: &[f32], alpha: f32, limit: f32) -> (usize, f64) {
    let (mut misses, mut worst) = (0, 0.0f64);
    for ((&v, &g), &u) in got.iter().zip(g).zip(u) {
        let want = oai_exact(g, u, alpha, limit);
        let band = OAI_REL * want.abs() + oai_flush(alpha, limit) + f64::from(f32::MIN_POSITIVE);
        let d = (f64::from(v) - want).abs() / band;
        misses += usize::from(d.is_nan() || d > 1.0);
        worst = worst.max(d);
    }
    (misses, worst)
}
