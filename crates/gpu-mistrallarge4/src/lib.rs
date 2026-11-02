//! bloomery-gpu-mistrallarge4 — the Mistral Large 4 program: a placed body
//! over the runtime's kernels, host code only. It has no device code of its
//! own: the norms, the LayerNorm, the rope, the SSMax pair, the fused key
//! append, the 576-wide latent flash and its selected row gather, the q8_0
//! gemvs and per-head absorption and the head are `bloomery-gpu`'s; the
//! lightning indexer, its input-projected key append, the selected 512-wide
//! walk and the common routed block (the softmax router, the card's expert
//! slots, the shared expert and the host leg's join)
//! `bloomery-gpu-deepseek41`'s. Built with `cargo oxide`, as the device
//! crates it links are.
//!
//! - [`body`]: the placed load (the host tier and the plan's expert tier
//!   card hung under it), the stores, the step's buffers, [`body::Body`]
//!   behind `GpuModel` and the gate instruments (taps, spans, the routing,
//!   the lists, the scores);
//! - [`features`]: the feature tap a block draft reads, the rows entering the layers its file
//!   names, laid by the step, a verify's rows and the prompt batch;
//! - [`ffn`]: the routed block's family facts (the router, the card experts,
//!   the residual order);
//! - [`prefill`]: the prompt batch, the common walk over units of rows with
//!   the attention this family's layers run over them;
//! - [`resid`]: the adaptive expert residency's facts (`BLOOMERY_RESIDENCY`) the common machine
//!   is built from, and the instruments that read its boundaries;
//! - `step`: one token's walk over the layers and the head;
//! - `tensors`: each layer's attention tensors, built from the file's own
//!   arrays;
//! - `tier`: the expert tier card's node service over the file's
//!   description ([`NODE_FAMILY`]), the common card leg's tier;
//! - [`stream`]: the prompt calls' host streaming (`BLOOMERY_XSTREAM`) the
//!   common walk runs, with the family's facts.
//!
//! Not this crate's (later rounds): DSpark, Pixtral.

pub mod body;
pub mod draft;
pub mod features;
pub mod ffn;
pub mod prefill;
pub mod resid;
mod step;
pub mod stream;
mod tensors;
mod tier;

pub use tier::NODE_FAMILY;
