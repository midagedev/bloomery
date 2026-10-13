//! The DSpark draft on its own card, V4.1's side of the common owner
//! ([`bloomery_gpu::dspark`]): the draft file's tensors and the two the
//! target lends it ([`load`]), the layers' context append ([`kv`]) and the
//! block pass over `[id_last, mask × (w − 1)]` ([`block`]), which
//! [`layers::V41Layers`] runs as the owner's
//! [`DraftLayers`](bloomery_gpu::dspark::DraftLayers). [`open`] moves fc, the
//! head and the Markov tables out of the weights into the owner's inputs and
//! builds [`DraftBody`], the owner over these layers: the sequence of calls,
//! the positions' rules and the graphs per shape are
//! [`DsparkBody`](bloomery_gpu::dspark::DsparkBody)'s.
//!
//! The draft is `model::arch::dspark`'s file: three window-only V4.1-shaped
//! blocks behind `fc`, which projects the target's hidden states at
//! `target_layers`. Everything here reuses kernels the target or the draft
//! kernel rounds already own; this module adds no `#[kernel]`.
//!
//! The feature of one committed target position is the mean of the four
//! hyper-connection streams leaving each target layer `target_layers[i] - 1`
//! (the input of layer `target_layers[i]`), the three means concatenated in
//! `target_layers` order: `hp.target_layers.len() · n_embd` f32 per position.
//!
//! # Positions
//!
//! The reference `model.py`'s rule (`DSparkAttention.forward`): the committed
//! row of target position `p` turns at rope position `p` and sits in ring
//! slot `p % window`; a block after the last committed position `p` turns
//! row `j` at `p + 1 + j`, and every block row sees the whole ring (its
//! first `min(p + 1, window)` rows) and every row of the block. ik's oracle
//! turns every committed row after the prompt at `p + 1` and its block one
//! further on; the gates feed an oracle set its own positions, the engine
//! never does.
//!
//! # Submission
//!
//! The owner captures at load one graph per block width and per append row
//! count, over the launches [`layers::V41Layers`] makes. The graphs are
//! captured under [`block::Rule::Reference`], the engine's rule: its clamp
//! limits are launch scalars, so a pass under [`block::Rule::Ik`] runs eager
//! ([`bloomery_gpu::dspark::Submit::Eager`], the gates' twin of every call),
//! and a graph replay under it is refused by name
//! ([`bloomery_gpu::dspark::DraftLayers::replayable`]).

pub mod block;
pub mod kv;
pub mod layers;
pub mod load;

use std::sync::Arc;

use bloomery_gpu::dspark::DsparkBody;
use bloomery_gpu::{Gpu, GpuError};
use gguf::Split;
use model::arch::dspark::shape;

pub use layers::V41Layers;
use load::DraftWeights;

/// V4.1's draft: the common owner over [`V41Layers`].
pub type DraftBody = DsparkBody<V41Layers>;

/// Open V4.1's draft on `gpu`: the owner's inputs moved out of `weights`
/// ([`load::owner_inputs`]), the layers over the rest (the target's file
/// `target` serves each block's first-id embedding row) and the owner over
/// both. Load-time only.
pub fn open(
    gpu: &Gpu,
    mut weights: DraftWeights,
    target: Arc<Split>,
) -> Result<DraftBody, GpuError> {
    let shape = shape(weights.hp());
    let (fc, head, embed) = load::owner_inputs(&mut weights)?;
    let layers = V41Layers::new(gpu, weights, target)?;
    DsparkBody::new(gpu, &shape, layers, fc, head, embed)
}
