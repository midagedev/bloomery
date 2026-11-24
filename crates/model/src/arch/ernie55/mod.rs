//! `ernie55` — Baidu's ERNIE 5.5 Flash Preview. The trunk is 43 layers on four hyper-connection
//! streams (mHC: a sinkhorn-normed 4x4 comb, the post mix scaled by `hyper_connection.magnitude`);
//! the first layer is a dense `situ` block and the rest route 10 of 512 experts that run in a
//! 2048-value latent beside one shared expert, the router the sum of two sigmoid projections of
//! the hidden row, the kept weights renormalized and scaled by a learned factor an expert. Two
//! kinds of mixer share the trunk, told apart by `attention.compress_ratios`:
//! - an HCA layer (ratio 128), DeepSeek V4.1's compressed attention: one key and value head of 512
//!   values, a 128-position window beside one pooled row every 128 positions (YaRN on the pooled
//!   rows), grouped output projection and a per-head sink;
//! - a latent layer (ratio 0, every seventh layer from the eighth), gated latent attention in the
//!   absorbed form (a 576-value cached row) whose positions a lightning indexer selects.
//!
//! Both gate their heads' outputs by a sigmoid projection of the normed query latent and mix the
//! heads' outputs by `I + V·Uᵀ` (`attn_vha_u`, `attn_vha_v`). The head is the untied matrix
//! behind a MultiMax epilogue of eight values (`output_multimax_*`). The file's keys
//! (`hparams`) and where the checkpoint's config keys go (`config`), every tensor's role (`roles`), the tensor names a step reads (`names`), what a
//! layer's stores are (`kv`), the description (`spec`), what it cannot state (`ops`) and the plan's
//! inputs (`place`).
//!
//! The dialect's authority is the model folder's `config.json` and checkpoint, with llama.cpp's
//! `src/models/ernie5-5.cpp` (PR #31925) and vLLM-ERNIE's `model.py`, `attention.py`, `moe.py`
//! and `mhc.py` for the graph.

pub mod config;
pub mod hparams;
pub mod kv;
pub mod names;
pub mod ops;
pub mod place;
pub mod roles;
pub mod spec;
