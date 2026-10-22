//! The pair's band and the fold's band.
//!
//! PIN(2026-10-23): pair = the norm-on-load mat-vec band, silu slope under 1.1, exp's 4 ulp; fold n = 13 + ceil(18/4) + 31 + 2.

use bloomery_metal::kernels::Quant;

use super::oracle::{self, Kind};

/// f32 unit roundoff, the value the mat-vec band's gamma is built from.
const U: f64 = 1.0 / 16_777_216.0;

/// Operations on the store path of a mat-vec of `nblocks` blocks.
fn store_ops(quant: Quant, nblocks: usize) -> usize {
    match quant {
        Quant::Q4K => 13 + nblocks.div_ceil(4) + 31,
        Quant::Q6K => 9 + nblocks.div_ceil(2) + 31,
    }
}

/// Operations on the pair's dot path. The pair norms its input in the
/// mat-vec's norm-on-load order: the sum of squares (a square and two adds per
/// four values, the lane's chain of `8 ceil(nb / 4)` adds for Q4_K or
/// `4 ceil(nb / 2)` for Q6_K, the SIMD group's 31), its division and addition,
/// then the store path with the gain product on the load and the scale
/// product at the end.
fn norm_ops(quant: Quant, nblocks: usize) -> usize {
    let chain = match quant {
        Quant::Q4K => 8 * nblocks.div_ceil(4),
        Quant::Q6K => 4 * nblocks.div_ceil(2),
    };
    let sum_squares = 1 + 2 + chain + 31;
    sum_squares + 2 + store_ops(quant, nblocks) + 2
}

/// Operations on the fold's path. Q4 is `13 + ceil(18/4) + 31 + 2`. Q6 is the
/// same count with its base and its stride: `9 + ceil(18/2) + 31 + 2`. The
/// added two are the route-weight product on the load and the residual add.
fn fold_ops(quant: Quant) -> usize {
    match quant {
        Quant::Q4K => FOLD_OPS_Q4,
        Quant::Q6K => FOLD_OPS_Q6,
    }
}

const FOLD_OPS_Q4: usize = 13 + 18_usize.div_ceil(4) + 31 + 2;
const FOLD_OPS_Q6: usize = 9 + 18_usize.div_ceil(2) + 31 + 2;
const _: [(); 51] = [(); FOLD_OPS_Q4];
const _: [(); 51] = [(); FOLD_OPS_Q6];

fn silu(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

/// `(want, width)` for one output `silu(g) * u`.
///
/// `g` and `u` are the f64 dots over the normed input and `mag_g`, `mag_u`
/// the sums of the term magnitudes. Each dot sits in the norm-on-load
/// mat-vec band `factor * mag * (1 + U)`.
/// silu's slope `σ(1 + x(1 − σ))` stays under 1.1, so that part of the error
/// is `1.1` times the gate's band. exp is taken to be off by 4 ulp, `8 U`
/// relative; `E / (1 + E)^2` is at most a quarter, which brings that term down
/// to `2 U` times the gate. The addition of one, the division and the rounding
/// of silu add `3 U (|g| + |silu|)`. The product with `u` keeps both first-order
/// terms, their product, and one rounding of the result.
pub fn pair_width(
    quant: Quant,
    nblocks: usize,
    g: f64,
    mag_g: f64,
    u: f64,
    mag_u: f64,
) -> Result<(f64, f64), String> {
    let dot = oracle::band(quant, nblocks, Kind::Norm);
    let ops = norm_ops(quant, nblocks);
    if dot.ops() != ops {
        return Err(format!(
            "the norm-on-load band is {} operations, the pair derives {ops}",
            dot.ops()
        ));
    }
    let g_err = dot.factor() * mag_g * (1.0 + U);
    let u_err = dot.factor() * mag_u * (1.0 + U);
    let s = silu(g);
    if !s.is_finite() {
        return Err(format!("silu of {g:e} is not finite"));
    }
    let exp_abs = 2.0 * U * (g.abs() + g_err);
    let round_abs = 3.0 * U * (g.abs() + s.abs());
    let silu_err = 1.1 * g_err + exp_abs + round_abs;
    let y = s * u;
    let prod = silu_err * u.abs() + s.abs() * u_err + silu_err * u_err + U * y.abs();
    let floor = (ops + 4) as f64 * f64::from(f32::MIN_POSITIVE);
    Ok((y, prod + floor))
}

/// The absolute band of a folded row. `y` is the f64 value, residual included,
/// and `magnitude` is the sum of the slots' term magnitudes. The residual
/// contributes `U |y|` on top of gamma of the fold's operation count.
#[must_use]
pub fn fold_width(quant: Quant, magnitude: f64, y: f64) -> f64 {
    let n = fold_ops(quant);
    let factor = oracle::gamma_bound(n);
    let floor = n as f64 * f64::from(f32::MIN_POSITIVE);
    factor * magnitude * (1.0 + U) + U * y.abs() + floor
}
