//! `hy_v4_flash`: Tencent's Hy4-flash, a text decoder of 42 layers and a next-token block. Every
//! layer is gated latent attention (MLA with a per-head sink and an output gate, 256-wide values,
//! a 576-row cache of the 512 latent and 64 rotated key values): the sparse-attention layers (32
//! heads, rope base 1e7) attend the tokens a pooled indexer picks, the window layers (64 heads,
//! rope base 1e4) the last 512 positions. The residual is four streams with identity-comb
//! hyper-connections and sandwich norms on both sublayers; the first layer is dense and the rest
//! route a sigmoid top-16 of 512 experts that run in a 1536-value latent beside one shared expert,
//! the routed gate clamped before the activation. Four layers also run an engram site before their
//! mixer: rows of one fused per-layer table gathered by a hash of the last three mapped tokens.
//! The file's keys (`hparams`), every tensor's role (`roles`), the tensor names a step reads
//! (`names`), the typed description (`spec`) and the by-position buffers the plan sizes (`place`).
//! The dialect's authority is llama.cpp's `src/models/hy-v4-flash.cpp` (PR #32517).

/// `general.architecture` of a Hy4-flash file, the registry's canonical name.
pub use gguf::arch::hy_v4_flash::ARCH;

pub mod fixture;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
