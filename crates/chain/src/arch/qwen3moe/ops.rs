//! The ops of one layer of a qwen3moe-family chain: what a backend enqueues for each part of a layer,
//! named once so [`super::walk`] orders them the same way for every backend.
//!
//! The order is the walk's; an op is the backend's. A layer is its mixer (attention over K/V
//! planes, or the delta rule over a recurrent store), then its FFN half (the router, then the
//! experts), and the unit ends in its head. Which kernel an op launches follows from the unit's
//! rows and the plan inside the op, and where the layer's output rows go (the residual stream, or
//! after the last layer what the unit ends in) from [`LayerCtx::last`] and the backend's own end
//! of the unit.

use super::plan::{DeltaPlan, FfnPlan, GqaPlan, LayerPlan};
use crate::ChainError;

/// What an op of one layer of a walk is given beside the plan part it runs: the layer, its plan, the
/// unit's rows, and whether the layer is the chain's last.
#[derive(Clone, Copy)]
pub struct LayerCtx<'p> {
    pub layer: usize,
    pub plan: &'p LayerPlan,
    /// The unit's rows.
    pub m: usize,
    /// The chain's last layer: its output rows are the last a unit's end reads.
    pub last: bool,
}

impl<'p> LayerCtx<'p> {
    /// Layer `layer` of the chain `plans` at `m` rows, or a named refusal for a layer the chain does
    /// not hold.
    pub fn of(plans: &'p [LayerPlan], layer: usize, m: usize) -> Result<LayerCtx<'p>, ChainError> {
        let plan = plans.get(layer).ok_or_else(|| {
            ChainError::shape(
                "qwen3moe::LayerCtx::of",
                format!("layer {layer} of a chain of {} layers", plans.len()),
            )
        })?;
        Ok(LayerCtx {
            layer,
            plan,
            m,
            last: layer + 1 == plans.len(),
        })
    }
}

/// What a backend enqueues for a layer's parts. Every op only enqueues: no synchronization, no
/// allocation, no host round trip.
pub trait ChainOps {
    /// A backend's refusal; a plan's own refusals convert into it.
    type Error: From<ChainError>;

    /// The unit's front, once, ahead of layer 0: the embedding rows of the unit's ids with each
    /// row's position and live key count, and what the backend's arm for `m` rows reads of the
    /// unit beside them.
    fn embed(&mut self, m: usize) -> Result<(), Self::Error>;

    /// The attention half's input side: the norm with its quantizer, then q, k and v.
    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), Self::Error>;

    /// The attention half's middle: the q/k norm and rope with the cache append, the flash over
    /// the rows' live keys, and the output projection's input (with the output gate at head 256).
    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), Self::Error>;

    /// The attention half's output side: the output projection and the residual add into the
    /// FFN's input.
    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), Self::Error>;

    /// The delta rule's input side: the norm with its quantizer and the four input projections.
    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), Self::Error>;

    /// The delta rule's middle: the conv and prep over the layer's ring, the delta step over its
    /// state, and the gated norm.
    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), Self::Error>;

    /// The delta rule's output side: the output projection and the residual add into the FFN's
    /// input.
    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), Self::Error>;

    /// The FFN half's front: the FFN norm with its quantizer, and the router's pick of every
    /// row's slots (a dense FFN has no pick: its one slot a row).
    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), Self::Error>;

    /// The FFN half's rest: the routed slots' gate·up·SwiGLU and down, the shared expert, and the
    /// combine into the layer's output rows.
    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), Self::Error>;

    /// What the unit ends in after its last layer: the head over the unit's rows, or the final
    /// norm of every row, or nothing when the last layer's output stays in the residual stream.
    fn head(&mut self, m: usize) -> Result<(), Self::Error>;
}
