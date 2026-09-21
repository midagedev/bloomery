//! `GpuModel` — the GPU engine that stands beside `forward::step`
//! (docs/gpu-design.md decision 1). Weights, KV and scratch live on the
//! device from `load`; `step` enqueues one decode step on the engine's stream
//! and synchronizes once for the argmax. P0 leaves the body a skeleton: the
//! buffer and stream ownership are real, the kernels behind `step` arrive with
//! P1–P7 and P8 assembles them.

use crate::{Gpu, GpuError};
use model::attn::MlaParams;

/// Compile-time probe of the dependency direction (gpu → model → gguf): the
/// device-bundle crate reads model metadata through `bloomery-model`.
/// Kept until `GpuModel::load` reads the same parameters for real.
pub fn mla_width(gguf: &gguf::Gguf) -> Result<usize, GpuError> {
    let p = MlaParams::read(gguf, 0)?;
    Ok(p.rope_dims + p.latent)
}

/// One resident model on one device. Owns its `Gpu` (context, stream,
/// module); everything `step` touches is allocated here, never per step.
pub struct GpuModel {
    gpu: Gpu,
    mla: MlaParams,
    /// KV rows the resident cache was sized for; `step` refuses to grow it.
    ctx_max: usize,
    /// Captured decode step, once P8 assembles one. `None` runs eagerly.
    graph: Option<crate::graph::Graph>,
}

impl GpuModel {
    /// Read the metadata `step` needs and size the resident buffers for
    /// `ctx_max` KV rows. Weight upload and KV/scratch allocation land here
    /// as P1–P7 deliver their formats; P0 holds only the parameters.
    pub fn load(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel, GpuError> {
        if ctx_max == 0 {
            return Err("GpuModel::load: ctx_max must be >= 1".into());
        }
        let gpu = Gpu::new()?;
        let mla = MlaParams::read(gguf, 0)?;
        Ok(GpuModel {
            gpu,
            mla,
            ctx_max,
            graph: None,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn mla(&self) -> &MlaParams {
        &self.mla
    }

    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    /// Whether `step` replays a captured graph (P8) or enqueues eagerly.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// One decode step: append `tokens` to the KV cache and return the argmax
    /// of the last position's logits. Not assembled before P8; until then
    /// this is an error, not a panic, so a caller can probe for it.
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        if tokens.is_empty() {
            return Err("GpuModel::step: empty token slice".into());
        }
        Err("GpuModel::step: not assembled yet (P8; kernels P1–P7)".into())
    }
}
