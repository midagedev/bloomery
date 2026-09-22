//! Synthetic KV cache contents: the deterministic row pattern and the
//! whole-cache write the depth and gate seeds go through.

use crate::tensor::DeviceTensor;
use crate::{Gpu, GpuError};

/// `rows` cache rows of `width` f16 values for [`GpuModel::seed_depth`](crate::GpuModel::seed_depth),
/// deterministic in `(rows, width)` alone.
///
/// An LCG over the flat index picks each value's sign and mantissa; the
/// exponent is one of two, so every magnitude lands in `[0.25, 1.0)` — a
/// real cache row's scale, and zero, inf and NaN are unreachable rather
/// than merely unlikely. Zero rows would flatten the softmax and NaN would
/// change what the causal guard means, so neither may be produced by
/// accident.
pub(super) fn seed_pattern(rows: usize, width: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(rows * width);
    let mut s: u32 = 12345;
    for _ in 0..rows * width {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        // f16 = sign(1) | exponent(5) | mantissa(10); exponent 13 gives
        // [0.25, 0.5) and 14 gives [0.5, 1.0).
        let sign = (s >> 24) & 1;
        let exponent = 13 + ((s >> 23) & 1);
        let mantissa = (s >> 13) & 0x3ff;
        let bits = (sign << 15) | (exponent << 10) | mantissa;
        out.push(u16::try_from(bits).expect("1 sign + 5 exponent + 10 mantissa bits"));
    }
    out
}

/// Write `rows` (whole cache rows) at the head of `cache`, zeroing the rest.
/// Synchronizes; never inside a capture.
pub(super) fn seed_cache(
    gpu: &Gpu,
    cache: &mut DeviceTensor<u16>,
    rows: &[u16],
    what: &'static str,
) -> Result<(), GpuError> {
    let full_len = cache.rows() * cache.cols();
    if rows.is_empty() || rows.len() > full_len || !rows.len().is_multiple_of(cache.cols()) {
        return Err(GpuError::shape(
            what,
            format!(
                "{} values are not whole {}-wide rows inside {full_len}",
                rows.len(),
                cache.cols()
            ),
        ));
    }
    let mut full = vec![0u16; full_len];
    full[..rows.len()].copy_from_slice(rows);
    cache.buf_mut().copy_from_host(gpu.stream(), &full)?;
    Ok(())
}
