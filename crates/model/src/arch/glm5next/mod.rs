//! `glm5next` — GLM-5.3-Flash, which a file declares as `glm5next` (ik) or
//! `glm5-next` (llama.cpp, [`crate::arch::GLM5_NEXT_LLAMA_CPP`]); its keys sit
//! under the file's own string. The file's keys (`hparams`), every tensor's
//! role (`roles`), the typed description (`spec`), the tensor names the
//! program reads (`names`), the host tier's view of a routed layer (`host`)
//! and the plan from the headers (`place`); the program is `gpu-glm5next`.
//!
//! The trunk interleaves KDA delta-rule layers with absorbed, rope-free
//! latent attention layers that pick their positions by a token-pool
//! indexer; every trunk block is wrapped in hyper-connection streams, which
//! collapse to their unweighted mean before the head. The next-token (MTP)
//! block follows the trunk: a latent layer with a plain residual. The
//! dialect's authority is ik_llama.cpp; the line numbers cited are its
//! `src/llama-hparams.cpp` unless another file is named.

pub mod fixture;
pub mod host;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
