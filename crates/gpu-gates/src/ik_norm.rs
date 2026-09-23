//! ik's RMS norm of one row on its CPU backend (`ggml_compute_forward_rms_norm_f32`,
//! and `ggml_compute_forward_fused_rms_norm_f32` for `FUSED_RMS_NORM`, which
//! multiplies in the gain): each value squared in f32, the squares summed
//! serially in f64, the mean rounded to f32, then `1/sqrtf(mean + eps)`. A
//! gate that simulates ik's norm reads the rule here; it is transcribed from
//! ik and calls no engine or device code.

/// ik's norm scale of row `x`: `1/sqrtf(mean + eps)` over the mean of its
/// squares.
pub fn scale(x: &[f32], eps: f32) -> f32 {
    let sum = x.iter().fold(0.0f64, |a, &v| a + f64::from(v * v));
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

/// ik's `FUSED_RMS_NORM` of row `x`: `(scale · gain) · x` per value, with
/// [`scale`]'s scale.
pub fn fused(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let s = scale(x, eps);
    x.iter().zip(gain).map(|(&v, &g)| (s * g) * v).collect()
}
