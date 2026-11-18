//! `raon1` — Raon-1-mini. The file's keys (`hparams`), every tensor's role (`roles`), the typed
//! description (`spec`), the tensor names a step reads (`names`), the file planned onto a
//! machine's cache bytes (`place`) and the gate fixture's spec (`fixture`).
//!
//! The trunk mixes with Raon delta attention (RDA) layers and gated latent-attention (MLA)
//! layers, the MLA layers being those whose `attention.head_count_kv` is 1 and the RDA layers
//! those whose is 0. An RDA layer is a Kimi delta rule whose erase gate `sigmoid(b)` and write
//! gate `sigmoid(b + w)` are apart, with a per-head RMS norm and a full-width sigmoid output gate.
//! An MLA layer rotates nothing: the `rope.dimension_count` key values beside the latent are
//! keys no rope turns, and its heads' outputs are gated by a projection of the layer input.
//! The first layers are dense `situ` blocks, the rest route by a sigmoid noaux_tc router over
//! `situ` experts beside an ungated shared one, and every layer's two sublayers mix the
//! residual's streams by Raon_mHC. The dialect's authority is llama.cpp's `src/models/raon1.cpp`.

pub mod fixture;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
