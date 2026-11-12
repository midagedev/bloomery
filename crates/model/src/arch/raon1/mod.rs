//! `raon1` — Raon-1-mini and Raon-1-nano, one reader of two dialects. The file's keys
//! (`hparams`), every tensor's role (`roles`), the typed description (`spec`), the tensor names a
//! step reads (`names`) and the file planned onto a machine's cache bytes (`place`).
//!
//! The trunk mixes with Raon delta attention (RDA) layers and gated latent-attention (MLA)
//! layers, the MLA layers being those whose `attention.head_count_kv` is 1 and the RDA layers
//! those whose is 0. An RDA layer is a Kimi delta rule whose erase gate `sigmoid(b)` and write
//! gate `sigmoid(b + w)` are apart, with a per-head RMS norm and a full-width sigmoid output gate.
//! An MLA layer rotates nothing: the `rope.dimension_count` key values beside the latent are
//! keys no rope turns, and its heads' outputs are gated by a projection of the layer input. The
//! Mini file's trunk is a `situ` dense lead over a sigmoid noaux_tc router beside an ungated
//! shared expert, its delta layers' projections split (`attn_q`/`attn_k`/`attn_v`, three convs)
//! and its head a matrix of its own; the Nano file's is dense SwiGLU blocks throughout with a
//! per-layer input gated into the streams after each, its delta layers' projections fused
//! (`attn_qkv`, one conv, the gates folded into `ssm_f_a`'s rows) and its head the token
//! embedding's, the facts its keys do not state the `raon1nano` row of `crate::arch::rules`.
//! Every layer's two sublayers mix the residual's streams by Raon_mHC. The dialects' authority
//! is llama.cpp's `src/models/raon1.cpp` and `src/models/raon1nano.cpp`.

pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
