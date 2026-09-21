//! Host-side reference for the GPU kernel gates (docs/gpu-design.md decision
//! 3, kernel layer). One owner for "what is the right answer" so the kernel
//! tracks do not each grow a generator: a weight row is dequantized with
//! `gguf::quant::dequant_row` — the scalar transcription gate-1-1 pins
//! against ggml's `to_float` — and dotted with the activation in f64. No
//! device code here; the gate binaries under `src/bin/` bring the device.
//!
//! The kernel gate is `max|y - y_ref| / max|y_ref| <= 1e-2` per shape, the
//! stage-0 contract for q8_1-activation kernels (measured floor 3–5e-3).

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, TensorInfo};

/// The model every gate reads unless `BLOOMERY_REF_MODEL` says otherwise.
pub const DEFAULT_MODEL: &str = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

/// The stage-0 kernel gate band.
pub const KERNEL_BAND: f32 = 1e-2;

pub type GateError = Box<dyn std::error::Error>;

pub fn open_model() -> Result<Gguf, GateError> {
    let path = std::env::var("BLOOMERY_REF_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    Ok(Gguf::open(path)?)
}

/// `m` activation columns of `k` f32 each, concatenated. A fixed LCG mapped
/// to [-1, 1) with every 61st value scaled by 8 so a block's amax is not
/// always near 1 — the quantizer's scale path sees spread. Values never
/// depend on time or on the host.
pub fn activations(k: usize, m: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    (0..k * m)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = ((s >> 8) & 0xffff) as f32 / 32_768.0 - 1.0;
            if i % 61 == 0 { u * 8.0 } else { u }
        })
        .collect()
}

/// Bytes of one row of `k` values of type `ty`.
pub fn row_bytes(ty: GgmlType, k: usize) -> Result<usize, GateError> {
    let blck = ty.blck_size().ok_or("row_bytes: unsupported type")?.max(1) as usize;
    let tsz = ty.type_size().ok_or("row_bytes: unsupported type")? as usize;
    if !k.is_multiple_of(blck) {
        return Err(format!("row_bytes: k = {k} is not a multiple of block {blck}").into());
    }
    Ok(k / blck * tsz)
}

/// Reference `y = W[row0 .. row0 + n_rows] · x` for raw rows of type `ty`
/// with `k` values each: `n_rows * m` f32, row-major with `m` outputs per
/// row (the kernels' output layout). `w` starts at row 0 of the span.
pub fn ref_gemv(
    ty: GgmlType,
    w: &[u8],
    k: usize,
    n_rows: usize,
    x: &[f32],
    m: usize,
) -> Result<Vec<f32>, GateError> {
    let rb = row_bytes(ty, k)?;
    if w.len() < rb * n_rows || x.len() < k * m {
        return Err(format!(
            "ref_gemv: w.len() {} < {} or x.len() {} < {}",
            w.len(),
            rb * n_rows,
            x.len(),
            k * m
        )
        .into());
    }
    let mut row = vec![0.0f32; k];
    let mut y = vec![0.0f32; n_rows * m];
    for r in 0..n_rows {
        dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut row)?;
        for c in 0..m {
            let xc = &x[c * k..(c + 1) * k];
            let dot: f64 = row.iter().zip(xc).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
            y[r * m + c] = dot as f32;
        }
    }
    Ok(y)
}

/// Raw bytes of tensor `name`, with its info (dims[0] is K, the row width).
pub fn tensor_bytes<'a>(gguf: &'a Gguf, name: &str) -> Result<(&'a TensorInfo, &'a [u8]), GateError> {
    let t = gguf.find(name).ok_or_else(|| format!("tensor {name} not in the model"))?;
    Ok((t, gguf.data(t)?))
}

/// `max|y - y_ref| / max|y_ref|`; an all-zero reference is an error, not 0.
pub fn max_rel_err(y: &[f32], y_ref: &[f32]) -> Result<f32, GateError> {
    if y.len() != y_ref.len() {
        return Err(format!("max_rel_err: len {} vs {}", y.len(), y_ref.len()).into());
    }
    let denom = y_ref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    if denom == 0.0 {
        return Err("max_rel_err: reference is all zero".into());
    }
    let num = y.iter().zip(y_ref).fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
    if !num.is_finite() {
        return Err("max_rel_err: non-finite kernel output".into());
    }
    Ok(num / denom)
}

/// Raw little-endian bytes as `u32` words (the K-quant kernels' load unit).
/// A length that is not a multiple of 4 is zero-padded in the last word.
pub fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}
