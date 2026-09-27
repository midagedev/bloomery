//! A host tier's computation over a run of routed layers ([`HostRun`]): each
//! layer's stacks as [`HostLayer`] reads them from the file, and the scratch
//! a step service writes, made at load. The architecture supplies only how
//! its layers' stacks are found (a closure over its names and widths, which
//! refuses a layer by its own error); the protocol around a call is the
//! tier's ([`super::HostTier`]).

use std::sync::Arc;

use gguf::Split;
use model::Tensor2;
use model::moe::{HostLayer, HostScratch};
use model::r8file::{R8Pair, R8Source};

use super::HostExperts;
use crate::GpuError;

/// The widths a host call serves: the model width, the routed experts'
/// feed-forward width and the routed slots a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostWidths {
    pub embd: usize,
    pub ff: usize,
    pub n_used: usize,
}

/// Every layer of the host run with its routed stacks, and the one-column
/// scratch a step service writes.
pub struct HostRun {
    /// The file and the r8 reading its layers were built from; every call
    /// reads through this pair.
    file: R8Pair,
    /// Per layer of the run, its host view.
    layers: Vec<HostLayer>,
    first: usize,
    scratch: HostScratch,
}

impl HostRun {
    /// The tier for the routed layers `first ..` of `file`, their host views
    /// made by `layers` from the file's reading — the architecture's names
    /// and widths, every refusal its own error. Load-time only: every stack
    /// is found and checked there, the gates and ups from `file`'s r8 sidecar
    /// when `r8` asks for it and there is one.
    pub fn build<E>(
        file: Arc<Split>,
        first: usize,
        r8: bool,
        widths: HostWidths,
        layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
    ) -> Result<HostRun, GpuError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let file = R8Pair::at_load(file, r8)?;
        let layers = layers(file.source()).map_err(|e| GpuError::Plan {
            what: "HostRun::build",
            source: Box::new(e),
        })?;
        Ok(HostRun {
            file,
            layers,
            first,
            scratch: HostScratch::new(widths.embd, widths.ff, widths.n_used)?,
        })
    }

    /// The first layer of the run.
    #[must_use]
    pub fn first(&self) -> usize {
        self.first
    }

    /// Layers in the run.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the run holds no layer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Layer `layer`'s host view, `None` outside the run.
    #[must_use]
    pub fn layer(&self, layer: usize) -> Option<&HostLayer> {
        layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
    }
}

impl HostExperts for HostRun {
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
                what: "HostRun::experts_into",
                missing: "the layer's routed stacks: it is outside the host run",
            })?;
        view.experts_into(self.file.source(), x, experts, out, &mut self.scratch)?;
        Ok(())
    }
}
