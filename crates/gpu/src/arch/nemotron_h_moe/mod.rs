//! The nemotron_h_moe family's half of the GPU engine: [`Body`], the decode
//! chain of Nemotron 3.5 Bolt — Mamba-2 layers, grouped-query attention layers
//! with no rope at sixteen query heads a key head, and latent MoE layers,
//! each MoE layer its own expert count and width (the description is
//! `model::arch::nemotron_h_moe`; the dialect's authority is llama.cpp's
//! `src/models/nemotron-h.cpp`).
//!
//! One layer is a pre-norm block and a residual add, by the layer's kind
//! (read from the file's per-layer arrays at load, not from its index):
//!
//! - Mamba-2 and attention: the hybrid trunk's layers
//!   ([`crate::arch::hybrid`]), the attention at the group-16 pass
//!   ([`crate::flash_gqa::GROUP_16`]); the projections here are NVFP4 (with
//!   the tensor's scale) or Q8_0;
//! - latent MoE: the router, the latent pair around the kept experts and the
//!   shared expert ([`moe`]'s module doc has the order).
//!
//! The token embedding is a Q8_0 table (`q38`'s `embed_rows_q8_0`) and the
//! output head a Q8_0 one ([`crate::head::Head`]). Every per-step value lives
//! in device words refreshed before the enqueue or replay, so one captured
//! graph serves every position. The body holds one sequence in one slot; a
//! prompt runs in passes of many rows ([`prefill`]). A load whose routed
//! experts the card cannot all hold keeps the plan's share resident and serves
//! the rest on the host tier ([`placed`]); its prompt runs in passes of up to
//! the host union's columns through the tier's batch port. The model is a
//! GGUF split or a checkpoint folder read as stored
//! ([`crate::folder::Model`]); a folder loads through that placed path, a plan
//! that leaves no expert on the host being its whole-card load.

mod body;
mod moe;
mod placed;
mod prefill;
pub mod router;

pub use crate::arch::hybrid::DecodeInput;
pub use body::Body;
pub use moe::Routing;
pub use placed::Placed;
pub use prefill::PASS_ROWS_MAX;
