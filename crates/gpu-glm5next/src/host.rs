//! The GLM host tier's computation ([`HostExperts`]): every routed layer's
//! stacks as [`HostLayer`] reads them from the file, and the scratch a
//! service writes, made at load. The protocol around it — the go, the wait,
//! the handoff's checks — is the host tier's own ([`bloomery_gpu::host`]).

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::GpuError;
use bloomery_gpu::hybrid::HostExperts;
use gguf::Split;
use model::Tensor2;
use model::arch::glm5next::host;
use model::arch::glm5next::hparams::Hparams;
use model::moe::{HostLayer, HostScratch};
use model::r8file::R8Pair;

/// Every layer of the host run with its routed stacks, and the one-column
/// scratch a step service writes.
pub struct GlmHost {
    /// The file and the r8 reading its layers were built from; every call
    /// reads through this pair.
    file: R8Pair,
    /// Per layer of the run, its host view.
    layers: Vec<HostLayer>,
    first: usize,
    scratch: HostScratch,
}

impl GlmHost {
    /// The tier for the routed layers `run` of `file`, whose hyperparameters
    /// are `hp`; a layer of the run that does not route is refused by name.
    /// Load-time only: every stack is found and checked here, the gates and
    /// ups from `file`'s r8 sidecar when `r8` asks for it and there is one.
    pub fn build(
        file: Arc<Split>,
        hp: &Hparams,
        run: Range<usize>,
        r8: bool,
    ) -> Result<GlmHost, GpuError> {
        let file = R8Pair::at_load(file, r8)?;
        let first = run.start;
        let layers = run
            .map(|l| {
                host::layer(file.source(), hp, l)?.ok_or(GpuError::State {
                    what: "GlmHost::build",
                    missing: "a routed layer's stacks: a layer of the host run does not route",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GlmHost {
            file,
            layers,
            first,
            scratch: HostScratch::new(hp.n_embd, hp.expert_ff, hp.n_used)?,
        })
    }
}

impl HostExperts for GlmHost {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let view = layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
            .ok_or(GpuError::State {
                what: "GlmHost::experts_into",
                missing: "the layer's routed stacks: it is outside the host run",
            })?;
        view.experts_into(self.file.source(), x, experts, out, &mut self.scratch)?;
        Ok(())
    }
}
