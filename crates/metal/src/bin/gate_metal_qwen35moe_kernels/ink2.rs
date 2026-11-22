//! What the Inkling 2 kernels' clause files share: the seeded draws, the
//! upload and read-back of a buffer, the error model's bands and the fit of a
//! kernel's output to its f64 oracle. Every clause of the family's kernels
//! runs on seeded synthetic inputs at the Mini's shapes with no fixture, so
//! each file states its oracle in f64 from the kernel's doc and its band from
//! the rounded operations the kernel's order performs.

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, Elem};
use bloomery_metal::MetalBuffer;

use super::Harness;
use super::oracle::Rng;

/// The unit roundoff of f32.
pub const U: f64 = 5.960_464_477_539_063e-8;

/// The relative error of `n` rounded operations, `n U / (1 - n U)`.
#[must_use]
pub fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// The relative error of `n` rounded operations when each rounding is
/// independent and at most `U`: they add as a random walk, `√n`, and the band
/// is three times that walk's step. The worst case [`gamma`] is the bound a
/// long dot is far under; this is the one a clause that must tell a mutant
/// one part in ten thousand from the kernel uses, named where it is.
#[must_use]
pub fn gamma_walk(n: usize) -> f64 {
    3.0 * U * (n as f64).sqrt()
}

/// Rounded operations of a dot of `n` terms a SIMD group sums: each lane's
/// `n / 32` in order, the butterfly's five levels and one more for the
/// rounding of the result.
#[must_use]
pub fn dot_ops(n: usize) -> usize {
    n.div_ceil(32) + 6
}

/// `n` values uniform in `[-scale, scale)`.
#[must_use]
pub fn draw(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| scale * rng.signed()).collect()
}

/// The buffer of `data`.
pub fn up<T: Elem>(h: &Harness<'_>, data: &[T]) -> Result<MetalBuffer<T>, String> {
    h.device.upload(h.stream, data).map_err(why)
}

/// The whole of `buffer`, read back.
pub fn read<T: Elem + Default>(h: &Harness<'_>, buffer: &MetalBuffer<T>) -> Result<Vec<T>, String> {
    let mut out = vec![T::default(); buffer.len().get()];
    h.device
        .read(h.stream, buffer.whole().map_err(why)?, &mut out)
        .map_err(why)?;
    Ok(out)
}

/// `n` words of the gate's sentinel NaN: an output the launch must write.
#[must_use]
pub fn blank(n: usize) -> Vec<f32> {
    vec![f32::from_bits(super::SENTINEL_BITS); n]
}

/// No element of `y` is still the sentinel.
pub fn written(what: &str, y: &[f32]) -> Result<(), String> {
    match y.iter().position(|v| v.to_bits() == super::SENTINEL_BITS) {
        Some(i) => Err(format!("{what}: element {i} was never written")),
        None => Ok(()),
    }
}

/// `got` against the oracle's `want` within `band` elementwise: the worst
/// error over its band (at most 1), refused by name at the first element past
/// it or not finite.
pub fn fit(what: &str, got: &[f32], want: &[f64], band: &[f64]) -> Result<f64, String> {
    if got.len() != want.len() || want.len() != band.len() {
        return Err(format!(
            "{what}: {} values against {} oracle values and {} bands",
            got.len(),
            want.len(),
            band.len()
        ));
    }
    let mut worst = 0.0f64;
    for (i, ((g, w), b)) in got.iter().zip(want).zip(band).enumerate() {
        if !g.is_finite() {
            return Err(format!("{what}: element {i} is {g:e}, the oracle's {w:e}"));
        }
        let err = (f64::from(*g) - w).abs();
        let ratio = if *b > 0.0 {
            err / b
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if ratio > 1.0 {
            return Err(format!(
                "{what}: element {i} is {g:e}, the oracle's {w:e}, off by {err:e} past a band of {b:e}"
            ));
        }
        worst = worst.max(ratio);
    }
    Ok(worst)
}

/// A refused launch: `result` is an error naming `needle`.
pub fn refused(what: &str, result: Result<(), String>, needle: &str) -> Result<(), String> {
    match result {
        Ok(()) => Err(format!("{what}: the launch was accepted")),
        Err(e) if e.contains(needle) => Ok(()),
        Err(e) => Err(format!(
            "{what}: refused as {e:?}, which does not name {needle:?}"
        )),
    }
}

/// `a` and `b` hold the same words.
pub fn same(what: &str, a: &[f32], b: &[f32]) -> Result<(), String> {
    match super::common::first_diff(
        &a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        &b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    ) {
        Some((i, x, y)) => Err(format!(
            "{what}: element {i} is {:e}, the clean run's {:e}",
            f32::from_bits(x),
            f32::from_bits(y)
        )),
        None => Ok(()),
    }
}
