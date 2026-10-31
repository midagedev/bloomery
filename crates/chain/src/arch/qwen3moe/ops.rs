//! The ops of one layer of a qwen3moe-family chain: what a backend enqueues for each part of a layer,
//! named once so [`super::walk`] orders them the same way for every backend.
//!
//! The order is the walk's; an op is the backend's. A layer is its mixer (attention over K/V
//! planes, the delta rule over a recurrent store, or the short conv over a conv ring), then its
//! FFN half (the router, then the experts), and the unit ends in its head. Which kernel an op
//! launches follows from the unit's rows and the plan inside the op, and where the layer's output
//! rows go (the residual stream, or after the last layer what the unit ends in) from
//! [`LayerCtx::last`] and the backend's own end of the unit.
//!
//! The join of a half's output into the stream has no op of its own: the op that ends each half
//! (`attn_out`, `delta_out` or `conv_out` for the mixer, `experts` for the FFN) reads
//! `l.plan.residual` and joins its output itself, `x += out` for a plain residual and
//! `x += add · gain ⊙ rms_norm(out)` for a sandwich (`ResidualPlan::Sandwich`). A backend that
//! joins by the plain add only calls [`LayerCtx::plain_only`] first in those ops, so a sandwich
//! layer stops at its first half with a named refusal and never runs as a plain one. An op a
//! backend has no kernels for has a default body that refuses by name, so a backend that does not
//! run a part of a layer stops at it and never runs the layer as another.

use super::plan::{DeltaPlan, FfnPlan, GqaPlan, LayerPlan, ResidualPlan, ShortConvPlan};
use crate::ChainError;
use std::any::type_name;

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

    /// `Ok` for a plain residual, and a named refusal of `op` for a sandwich: what a backend that
    /// joins a half's output by the plain add calls first in each op that ends a half.
    pub fn plain_only(&self, op: &'static str) -> Result<(), ChainError> {
        match &self.plan.residual {
            ResidualPlan::Plain => Ok(()),
            ResidualPlan::Sandwich(_) => Err(ChainError::shape(
                op,
                format!(
                    "layer {}: a sandwich residual, which this backend does not run",
                    self.layer
                ),
            )),
        }
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

    /// The attention half's input side: the norm with its quantizer, then q, k and v (and, on a
    /// kind with a gate of its own, the gate's projection of the normed rows).
    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), Self::Error>;

    /// The attention half's middle: the q/k norm and rope (by the plan's frequency factors, where
    /// it has them) with the cache append, the flash over the rows' live keys (the plan's window
    /// of them, where it has one), and the output projection's input (with the output gate where
    /// the kind has one).
    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), Self::Error>;

    /// What [`attn_core`](ChainOps::attn_core) runs first and nothing after it: the q/k norm and
    /// rope with the cache append, the store walk's last op ([`super::walk::store`]). A backend
    /// with no such launch keeps the default body, which refuses by name.
    fn attn_append(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), Self::Error> {
        Err(unrun::<Self>("qwen3moe::ChainOps::attn_append", l.layer).into())
    }

    /// The attention half's output side: the output projection and the join of its output into
    /// the FFN's input rows by `l.plan.residual` (the plain add, or the sandwich's mixer
    /// norm and scale).
    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), Self::Error>;

    /// The delta rule's input side: the norm with its quantizer and the four input projections.
    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), Self::Error>;

    /// The delta rule's middle: the conv and prep over the layer's ring, the delta step over its
    /// state, and the gated norm.
    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), Self::Error>;

    /// The delta rule's output side: the output projection and the join of its output into the
    /// FFN's input rows by `l.plan.residual` (the plain add, or the sandwich's mixer norm
    /// and scale).
    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), Self::Error>;

    /// The short conv's input side: the norm with its quantizer and the input projection of the
    /// normed rows to the three chunks `B | C | x` of `hidden` values each. A backend with no
    /// short-conv kernels keeps the default body, which refuses by name.
    fn conv_in(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), Self::Error> {
        Err(unrun::<Self>("qwen3moe::ChainOps::conv_in", l.layer).into())
    }

    /// The short conv's middle: `B ⊙ x`, the causal depthwise conv over the layer's ring and the
    /// rows' predecessors, and `C ⊙` the conv's output, the input of the output projection. A
    /// backend with no short-conv kernels keeps the default body, which refuses by name.
    fn conv_core(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), Self::Error> {
        Err(unrun::<Self>("qwen3moe::ChainOps::conv_core", l.layer).into())
    }

    /// The short conv's output side: the output projection and the join of its output into the
    /// FFN's input rows by `l.plan.residual` (the plain add, or the sandwich's mixer norm
    /// and scale). A backend with no short-conv kernels keeps the default body, which refuses by
    /// name.
    fn conv_out(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), Self::Error> {
        Err(unrun::<Self>("qwen3moe::ChainOps::conv_out", l.layer).into())
    }

    /// The FFN half's front: the FFN norm with its quantizer, and the router's pick of every
    /// row's slots (a dense FFN has no pick: its one slot a row).
    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), Self::Error>;

    /// The FFN half's rest: the routed slots' gate·up·SwiGLU and down, the shared expert, and the
    /// combine into the layer's output rows, joined to the half's input rows by `l.plan.residual`
    /// (the plain add, or the sandwich's FFN norm and scale).
    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), Self::Error>;

    /// What the unit ends in after its last layer: the head over the unit's rows, or the final
    /// norm of every row, or nothing when the last layer's output stays in the residual stream.
    fn head(&mut self, m: usize) -> Result<(), Self::Error>;
}

/// The refusal a default op body returns: the op, the layer and the backend (`B`'s type name)
/// that has no kernels for it.
fn unrun<B: ?Sized>(op: &'static str, layer: usize) -> ChainError {
    ChainError::shape(
        op,
        format!(
            "layer {layer}: the backend {} implements no such op",
            type_name::<B>()
        ),
    )
}
