//! The nemotron_h_moe family's chain plan: Nemotron 3.5 Bolt and Lightning, whose layers are
//! Mamba-2, NoPE attention or a latent mixture of routed experts.
//!
//! The mixers' plans, ops and walk are the hybrid trunk's ([`crate::hybrid`]); [`plan`] is the
//! family's own feed-forward layer and [`read`] makes the chain's plans from the file's header.

pub mod plan;
pub mod read;
