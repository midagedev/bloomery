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
