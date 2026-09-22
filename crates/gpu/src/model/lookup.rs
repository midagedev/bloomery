//! Resident weights by name, each accessor naming the variant it expects.

use crate::GpuError;
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use cuda_core::DeviceBuffer;

/// The resident weight by name — the `DevWeight` itself, which carries the
/// quantization type and `k` the byte accounting needs.
pub(super) fn dev_weight<'a>(w: &'a Weights, name: &str) -> Result<&'a DevWeight, GpuError> {
    w.get(name)
        .ok_or_else(|| GpuError::tensor("dev_weight", name, "resident"))
}

/// The resident derived q_nope2 planes named by `name` (`qs` rows x k/4
/// code words, `d` rows x k/32 scales, rows = n_head * latent, k = nope).
/// The name is `LayerNames::derived`, built at load.
pub(crate) fn q8_derived<'a>(
    w: &'a Weights,
    name: &str,
) -> Result<(&'a DeviceTensor<u32>, &'a DeviceTensor<f32>), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0Derived { qs, d, .. }) => Ok((qs, d)),
        Some(_) => Err(GpuError::tensor("q8_derived", name, "the derived variant")),
        None => Err(GpuError::tensor("q8_derived", name, "resident")),
    }
}

/// The resident Q3_K/Q4_K/Q6_K/Q5 word plane of a weight, by name.
pub(crate) fn kq_weight<'a>(w: &'a Weights, name: &str) -> Result<&'a DeviceTensor<u32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::KQuant { w, .. })
        | Some(DevWeight::Q5_0 { w, .. })
        | Some(DevWeight::Q5_1 { w, .. }) => Ok(w),
        Some(_) => Err(GpuError::tensor("kq_weight", name, "a word-plane variant")),
        None => Err(GpuError::tensor("kq_weight", name, "resident")),
    }
}

/// The resident F32 plane (norm gains), by name.
pub(crate) fn f32_gain<'a>(w: &'a Weights, name: &str) -> Result<&'a DeviceBuffer<f32>, GpuError> {
    Ok(f32_tensor(w, name)?.buf())
}

/// The resident F32 plane of a weight as a tensor — the router matrix, whose
/// gemv addresses rows.
pub(crate) fn f32_tensor<'a>(
    w: &'a Weights,
    name: &str,
) -> Result<&'a DeviceTensor<f32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w, .. }) => Ok(w),
        Some(_) => Err(GpuError::tensor("f32_tensor", name, "F32")),
        None => Err(GpuError::tensor("f32_tensor", name, "resident")),
    }
}
