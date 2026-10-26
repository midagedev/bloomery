//! The nemotron_h family's chain plan: Nemotron-H Nano, whose layers are Mamba-2, NoPE attention or
//! a dense squared-ReLU MLP.
//!
//! The mixers' plans, ops and walk are the hybrid trunk's ([`crate::arch::hybrid`]); [`plan`] is
//! the family's own feed-forward layer and [`read`] makes the chain's plans from the file's header.

pub mod plan;
pub mod read;
