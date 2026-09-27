//! `glm5next` — GLM-5.3-Flash. A header reader only: no program here runs
//! this architecture yet. The file's keys (`hparams`), every tensor's role
//! (`roles`) and the typed description (`spec`) exist so the coverage check
//! can list what a program has to run.
//!
//! The trunk interleaves KDA delta-rule layers with absorbed, rope-free
//! latent attention layers that pick their positions by a token-pool
//! indexer; every trunk block is wrapped in hyper-connection streams, which
//! collapse to their unweighted mean before the head. The next-token (MTP)
//! block follows the trunk: a latent layer with a plain residual. The
//! dialect's authority is ik_llama.cpp; the line numbers cited are its
//! `src/llama-hparams.cpp` unless another file is named.

pub mod hparams;
pub mod roles;
pub mod spec;
