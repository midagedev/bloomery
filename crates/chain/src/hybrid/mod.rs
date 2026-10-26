//! The Mamba-2 and no-rope attention hybrid's chain, the trunk Nemotron-H's families share: each
//! layer is exactly one of a Mamba-2 mixer, a grouped-query attention with no rope, or the
//! family's feed-forward block.
//!
//! [`plan`] is what a layer of either mixer is made of; [`read`] makes it from the file's header.
//! [`ops`] names what a backend enqueues for each part of a layer, and [`walk`] orders them. The
//! family's own feed-forward plan (the dense MLP, the latent MoE) lives with the family and enters
//! the walk as [`ops::HybridOps::Ffn`].

pub mod ops;
pub mod plan;
pub mod read;
#[cfg(test)]
pub(crate) mod test_file;
pub mod walk;
