//! The ops of one layer of a hybrid chain: what a backend enqueues for each part of a layer, named
//! once so [`super::walk`] orders them the same way for every backend.
//!
//! The order is the walk's; an op is the backend's. A layer is one of three kinds and runs its
//! kind's three ops in order: the input side (the norm with its quantizer, then the projections in),
//! the middle (the kind's own work) and the output side (the projection out and the residual add,
//! which ends the layer). Which buffer a layer reads and writes, and what a state lane is, are the
//! backend's; the ops are told the layer, the unit's rows and whether the layer is the chain's last.

use super::plan::{MambaPlan, NopeGqaPlan};
use crate::ChainError;

/// What an op of one layer is told beside the plan part it runs: the layer, the unit's rows, and
/// whether the layer is the chain's last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub layer: usize,
    /// The unit's rows.
    pub m: usize,
    /// The chain's last layer: its output rows are the last a unit's head reads.
    pub last: bool,
}

/// What a backend enqueues for a layer's parts. Every op only enqueues: no synchronization, no
/// allocation, no host round trip.
pub trait HybridOps {
    /// A backend's refusal; a plan's own refusals convert into it.
    type Error: From<ChainError>;

    /// The family's feed-forward plan: the dense MLP's, or the routed experts'.
    type Ffn;

    /// The unit's front, once, ahead of layer 0: the embedding rows of the unit's ids with each
    /// row's position and live key count.
    fn embed(&mut self, m: usize) -> Result<(), Self::Error>;

    /// A Mamba-2 layer's input side: the norm with its quantizer, then `ssm_in`, whose output is
    /// the `z`, `xBC` and `dt` views the scan reads.
    fn mamba_in(&mut self, s: Step, p: &MambaPlan) -> Result<(), Self::Error>;

    /// A Mamba-2 layer's middle: the conv over the layer's ring with its bias and SiLU, the softplus
    /// of `dt`, the scan over the layer's state, the `D` skip, the `z` gate and the grouped RMS norm.
    fn mamba_core(&mut self, s: Step, p: &MambaPlan) -> Result<(), Self::Error>;

    /// A Mamba-2 layer's output side: `ssm_out` and the residual add; it ends the layer.
    fn mamba_out(&mut self, s: Step, p: &MambaPlan) -> Result<(), Self::Error>;

    /// An attention layer's input side: the norm with its quantizer, then q, k and v.
    fn attn_in(&mut self, s: Step, p: &NopeGqaPlan) -> Result<(), Self::Error>;

    /// An attention layer's middle: the K and V appends at each row's position, and the flash over
    /// the rows' live keys.
    fn attn_core(&mut self, s: Step, p: &NopeGqaPlan) -> Result<(), Self::Error>;

    /// An attention layer's output side: the output projection and the residual add; it ends the
    /// layer.
    fn attn_out(&mut self, s: Step, p: &NopeGqaPlan) -> Result<(), Self::Error>;

    /// A feed-forward layer's input side: the norm, and what the family computes before its
    /// activation.
    fn ffn_in(&mut self, s: Step, f: &Self::Ffn) -> Result<(), Self::Error>;

    /// A feed-forward layer's middle: the family's activation.
    fn ffn_core(&mut self, s: Step, f: &Self::Ffn) -> Result<(), Self::Error>;

    /// A feed-forward layer's output side: the projection out and the residual add; it ends the
    /// layer.
    fn ffn_out(&mut self, s: Step, f: &Self::Ffn) -> Result<(), Self::Error>;

    /// What the unit ends in after its last layer: the final norm and the head over the unit's
    /// rows.
    fn head(&mut self, m: usize) -> Result<(), Self::Error>;
}
