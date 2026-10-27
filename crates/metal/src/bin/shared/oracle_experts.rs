//! The pair's band and the fold's band.

use bloomery_metal::geom::ExpertsGeom;
use bloomery_metal::kernels::Quant;

use super::oracle::{self, Kind, U, silu};

/// The blocks of one fold in `quant`'s own blocks: the hidden states of
/// every slot, `slots * k` columns (nine slots of 512 columns are eighteen
/// 256-value blocks, seventy-two A4 groups, a hundred and forty-four Q8_0
/// blocks).
fn fold_blocks(e: &ExpertsGeom, quant: Quant) -> usize {
    e.slots * e.k / quant.values_per_block()
}

/// The operations the fold adds to a store path of [`fold_blocks`] blocks:
/// the route-weight product on the load and the residual add (2), over the
/// store path's own base and SIMD-group sum (31).
///
/// PIN(2026-10-23): pair = the norm-on-load mat-vec band, silu's slope under
/// 1.1, exp's 4 ulp; fold = the store path of the quant's own block count
/// plus the route weight and the residual (2). The model in this module's
/// doc, and no measured error, sets the counts.
/// PIN(2026-10-25): each quant's own block count widens A4's fold band from `G(45)` to `G(51)`.
const FOLD_EXTRA_OPS: usize = 2;

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
    let ops = dot.ops();
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
pub fn fold_width(e: &ExpertsGeom, quant: Quant, magnitude: f64, y: f64) -> f64 {
    let n = oracle::band(quant, fold_blocks(e, quant), Kind::Store).ops() + FOLD_EXTRA_OPS;
    let factor = oracle::gamma_bound(n);
    let floor = n as f64 * f64::from(f32::MIN_POSITIVE);
    factor * magnitude * (1.0 + U) + U * y.abs() + floor
}

/// [`fold_width`] for a down stack of `quant` and a shared down of `shared`:
/// both ranges add into one lane sum, so a term's path is bounded by its
/// range's end to end with the other's.
///
/// PIN(2026-10-27): the model in this module's doc, and no measured error,
/// sets the counts: the two store paths of the quants' own block counts, the
/// route weight and the residual (2).
#[must_use]
pub fn fold_width_apart(
    e: &ExpertsGeom,
    quant: Quant,
    shared: Quant,
    magnitude: f64,
    y: f64,
) -> f64 {
    let routed = oracle::band(
        quant,
        (e.slots - 1) * e.k / quant.values_per_block(),
        Kind::Store,
    )
    .ops();
    let shared = oracle::band(shared, e.k / shared.values_per_block(), Kind::Store).ops();
    let n = routed + shared + FOLD_EXTRA_OPS;
    let factor = oracle::gamma_bound(n);
    let floor = n as f64 * f64::from(f32::MIN_POSITIVE);
    factor * magnitude * (1.0 + U) + U * y.abs() + floor
}
