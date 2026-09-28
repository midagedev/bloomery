//! The GLM host tier's computation ([`HostExperts`]): every routed layer's
//! stacks as [`HostLayer`] reads them from the file, the scratch a step
//! service writes, made at load, and the union scratch a prompt batch's
//! service writes, made once a prompt asks for batches
//! ([`GlmHost::prepare_union`]). The protocol around it — the go, the wait,
//! the handoff's checks, the batch port — is the host tier's own
//! ([`bloomery_gpu::host`]).

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::GpuError;
use bloomery_gpu::hybrid::HostExperts;
use gguf::Split;
use model::arch::glm5next::host;
use model::arch::glm5next::hparams::Hparams;
use model::moe::{HostLayer, HostScratch, UnionScratch};
use model::r8file::R8Pair;
use model::{Tensor2, Tensor2View};

/// Every layer of the host run with its routed stacks, the one-column
/// scratch a step service writes, and a batch service's union scratch.
pub struct GlmHost {
    /// The file and the r8 reading its layers were built from; every call
    /// reads through this pair.
    file: R8Pair,
    /// Per layer of the run, its host view.
    layers: Vec<HostLayer>,
    first: usize,
    scratch: HostScratch,
    /// The widths the union scratch is made for: the model's, the routed
    /// experts' and the routed slots a token.
    widths: (usize, usize, usize),
    /// The union's slabs, once [`GlmHost::prepare_union`] made them.
    union: Option<UnionScratch>,
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
            widths: (hp.n_embd, hp.expert_ff, hp.n_used),
            union: None,
        })
    }

    /// The union's slabs for batch services of up to `cols` columns, made
    /// once: a load that never feeds a prompt in batches never holds them.
    /// A width past what was made is refused by name at the call.
    pub fn prepare_union(&mut self, cols: usize) -> Result<(), GpuError> {
        if self.union.is_none() {
            let (embd, ff, n_used) = self.widths;
            self.union = Some(UnionScratch::new_routed(embd, ff, cols, n_used)?);
        }
        Ok(())
    }
}

/// Layer `layer`'s host view in `layers`, the run from `first`, refused by
/// name (as `what`'s error) outside it.
fn view<'a>(
    layers: &'a [HostLayer],
    first: usize,
    what: &'static str,
    layer: usize,
) -> Result<&'a HostLayer, GpuError> {
    layer
        .checked_sub(first)
        .and_then(|i| layers.get(i))
        .ok_or(GpuError::State {
            what,
            missing: "the layer's routed stacks: it is outside the host run",
        })
}

impl HostExperts for GlmHost {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let view = view(&self.layers, self.first, "GlmHost::experts_into", layer)?;
        view.experts_into(self.file.source(), x, experts, out, &mut self.scratch)?;
        Ok(())
    }

    fn experts_union_into(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "GlmHost::experts_union_into";
        let view = view(&self.layers, self.first, WHAT, layer)?;
        let scratch = self.union.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the union's slabs (GlmHost::prepare_union)",
        })?;
        view.experts_union_into(self.file.source(), x, lists, out, scratch)?;
        Ok(())
    }
}
