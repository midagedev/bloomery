//! The relative errors the gates hold a tensor to its reference by: the L2 form [`rel`] and the
//! max-norm form [`rel_max`], each refusing a length mismatch or a non-finite value as infinity, so
//! none passes a band.

/// `‖got − want‖ / ‖want‖` in f64, normalised by `want`; infinite on a NaN or
/// an infinity in either vector or a length mismatch, so none passes a band.
pub fn rel(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += f64::from(y).powi(2);
    }
    let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
    if r.is_nan() { f64::INFINITY } else { r }
}

/// `max|got − want| / max|want|` in f64, the max-norm of [`rel`]: the
/// numerator alone when `want` is all zeros; infinite on a non-finite value
/// or a length mismatch.
pub fn rel_max(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let den = want.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let num = got.iter().zip(want).fold(0.0f64, |m, (&x, &y)| {
        m.max((f64::from(x) - f64::from(y)).abs())
    });
    if den > 0.0 { num / den } else { num }
}

#[cfg(test)]
mod tests {
    use super::{rel, rel_max};

    #[test]
    fn rel_is_zero_on_equal_and_infinite_on_a_hole() {
        let a = [1.0f32, -2.0, 3.0];
        assert_eq!(rel(&a, &a), 0.0);
        assert_eq!(rel_max(&a, &a), 0.0);
        let want = [3.0f32, 4.0];
        assert_eq!(rel(&[0.0, 0.0], &want), 1.0);
        assert!(rel(&a, &a[..2]).is_infinite());
        assert!(rel_max(&a, &a[..2]).is_infinite());
        assert!(rel(&a[..2], &a).is_infinite());
        for bad in [f32::NAN, f32::INFINITY] {
            let got = [1.0f32, bad, 3.0];
            assert!(rel(&got, &a).is_infinite());
            assert!(rel(&a, &got).is_infinite());
            assert!(rel_max(&got, &a).is_infinite());
            assert!(rel_max(&a, &got).is_infinite());
        }
    }

    #[test]
    fn rel_max_is_the_max_norm_over_want() {
        // max|got - want| = 1.5, max|want| = 4.
        assert_eq!(rel_max(&[1.0, 2.5], &[1.0, 4.0]), 0.375);
        // An all-zero want leaves the numerator.
        assert_eq!(rel_max(&[0.25, -0.5], &[0.0, 0.0]), 0.5);
        // The L2 form normalises by want's L2 norm: sqrt(0.25) / 5.
        assert!((rel(&[3.5, 4.0], &[3.0, 4.0]) - 0.1).abs() < 1e-12);
    }
}
