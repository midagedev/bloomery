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

#[cfg(test)]
mod tests {
    use super::*;

    /// √32 is the crest's bound, and the model's value is the one the
    /// bands' derivations quote.
    #[test]
    fn the_q8_32_error_model() {
        assert!((CREST_MAX_32 * CREST_MAX_32 - 32.0).abs() < 1e-12);
        assert!((q8_32_rel() - 1.2858e-2).abs() < 1e-6, "{}", q8_32_rel());
    }
}
