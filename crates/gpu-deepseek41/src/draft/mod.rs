//! The DSpark draft on its own card: the draft file's tensors and the two
//! the target lends it ([`load`]), and the feature-to-KV graph that turns the
//! target's committed hidden states into each draft layer's window ring
//! ([`kv`]).
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

pub mod kv;
pub mod load;
