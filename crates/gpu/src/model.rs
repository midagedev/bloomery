//! `GpuModel` — the GPU engine that stands beside `forward::step`
//! (docs/gpu-design.md decisions 1 and 7). Weights, KV and scratch live on
//! the device from `load`, split into stages by layer range; `step` enqueues
//! one decode step stage by stage and synchronizes once for the argmax. P0 leaves the body a skeleton: the
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

/// A contiguous range of blocks resident on one device (docs/gpu-design.md
/// decision 7). A stage owns its `Gpu` — context, stream, modules — and, as
/// P1–P7 land, its weights, KV rows and scratch; what crosses a stage
/// boundary is one hidden vector. Two stages may sit on the same card: that
/// is the shape the 2-stage = 1-stage bit-identity gate runs in.
pub struct Stage {
    gpu: Gpu,
    /// Blocks `layers.start..layers.end` of the model, in order.
    layers: std::ops::Range<usize>,
    /// Captured decode step of this stage, once P8 assembles one.
    graph: Option<crate::graph::Graph>,
}

impl Stage {
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layers.clone()
    }

    /// Whether this stage replays a captured graph or enqueues eagerly.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }
}

/// One resident model: its stages in layer order, covering every block
/// exactly once. Everything `step` touches is allocated at load, never per
/// step.
pub struct GpuModel {
    stages: Vec<Stage>,
    mla: MlaParams,
    /// KV rows the resident cache was sized for; `step` refuses to grow it.
    ctx_max: usize,
}

impl GpuModel {
    /// The whole model as one stage.
    pub fn load(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel, GpuError> {
        GpuModel::load_staged(gguf, ctx_max, &[])
    }

    /// Split the blocks at `cuts` (strictly ascending, each in
    /// `1..block_count`): `cuts.len() + 1` stages. Reads the metadata `step`
    /// needs and sizes the resident buffers for `ctx_max` KV rows; weight
    /// upload and KV/scratch allocation land here as P1–P7 deliver their
    /// formats.
    pub fn load_staged(
        gguf: &gguf::Gguf,
        ctx_max: usize,
        cuts: &[usize],
    ) -> Result<GpuModel, GpuError> {
        if ctx_max == 0 {
            return Err("GpuModel::load: ctx_max must be >= 1".into());
        }
        let n_layers = gguf
            .block_count()
            .ok_or("GpuModel::load: metadata key block_count missing")? as usize;
        let mut bounds = vec![0usize];
        for &c in cuts {
            if c <= *bounds.last().unwrap_or(&0) || c >= n_layers {
                return Err(format!(
                    "GpuModel::load_staged: cuts must ascend strictly inside 1..{n_layers}, got {cuts:?}"
                )
                .into());
            }
            bounds.push(c);
        }
        bounds.push(n_layers);
        let mut stages = Vec::with_capacity(bounds.len() - 1);
        for w in bounds.windows(2) {
            stages.push(Stage {
                gpu: Gpu::new()?,
                layers: w[0]..w[1],
                graph: None,
            });
        }
        let mla = MlaParams::read(gguf, 0)?;
        Ok(GpuModel {
            stages,
            mla,
            ctx_max,
        })
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    pub fn mla(&self) -> &MlaParams {
        &self.mla
    }

    pub fn ctx_max(&self) -> usize {
        self.ctx_max
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
