//! A host tier's computation over a run of routed layers ([`HostRun`]): each
//! layer's stacks as [`HostLayer`] reads them from the file, the scratch a
//! step service writes, made at load, and the union scratch a batch service
//! writes, made once a caller asks for batches ([`HostRun::prepare_union`]). The architecture supplies only how
//! its layers' stacks are found (a closure over its names and widths, which
//! refuses a layer by its own error); the protocol around a call is the
//! tier's ([`super::HostTier`]).

use std::sync::Arc;

use gguf::Split;
use model::moe::{HostLayer, HostScratch, UNION_MAX_COLS, UnionScratch};
use model::r8file::{R8Pair, R8Source};
use model::{Tensor2, Tensor2View};

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

/// Every layer of the host run with its routed stacks, the one-column
/// scratch a step service writes, and a batch service's union scratch.
pub struct HostRun {
    /// The file and the r8 reading its layers were built from; every call
    /// reads through this pair.
    file: R8Pair,
    /// Per layer of the run, its host view.
    layers: Vec<HostLayer>,
    first: usize,
    scratch: HostScratch,
    widths: HostWidths,
    /// The union's slabs, once [`HostRun::prepare_union`] made them.
    union: Option<UnionScratch>,
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
            widths,
            union: None,
        })
    }

    /// The union's slabs for batch calls of up to `cols` columns of the
    /// run's routed width, made once: a load that never serves a batch never
    /// holds them. Past [`UNION_MAX_COLS`] they are a group tail's
    /// ([`UnionScratch::new_tail`]) over whole batches of that many columns,
    /// with room for every slot, so a call of `cols` columns runs as one. A
    /// later call for no more columns than they hold keeps them; one for more
    /// is refused by name (the slabs are the load's size). Load-time only.
    pub fn prepare_union(&mut self, cols: usize) -> Result<(), GpuError> {
        match &self.union {
            None => {
                let w = self.widths;
                self.union = Some(if cols <= UNION_MAX_COLS {
                    UnionScratch::new_routed(w.embd, w.ff, cols, w.n_used)?
                } else {
                    let groups = cols.div_ceil(UNION_MAX_COLS);
                    let slots = groups * UNION_MAX_COLS * w.n_used;
                    UnionScratch::new_tail(w.embd, w.ff, groups, slots, w.n_used)?
                });
                Ok(())
            }
            Some(u) if cols <= u.max_cols() => Ok(()),
            Some(u) => Err(GpuError::shape(
                "HostRun::prepare_union",
                format!(
                    "slabs for {cols} columns; the load made them for {}",
                    u.max_cols()
                ),
            )),
        }
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

    fn experts_union_into(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let HostRun {
            file,
            layers,
            first,
            union,
            ..
        } = self;
        let view = layer
            .checked_sub(*first)
            .and_then(|i| layers.get(i))
            .ok_or(GpuError::State {
                what: "HostRun::experts_union_into",
                missing: "the layer's routed stacks: it is outside the host run",
            })?;
        let scratch = union.as_mut().ok_or(GpuError::State {
            what: "HostRun::experts_union_into",
            missing: "the union's slabs (HostRun::prepare_union)",
        })?;
        view.experts_union_into(file.source(), x, lists, out, scratch)?;
        Ok(())
    }
}
