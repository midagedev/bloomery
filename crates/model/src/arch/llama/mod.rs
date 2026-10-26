//! `llama` — the dense Llama architecture (grouped-query attention on the
//! pairwise rope, a SwiGLU feed-forward block, an untied head). The host code
//! that knows it: its hyperparameters (`hparams`), the role of every tensor
//! (`roles`) and the typed model description (`spec`). The tensor names are
//! the ones every family spells alike ([`crate::arch::names`]).

pub mod hparams;
pub mod roles;
pub mod spec;

#[cfg(test)]
mod nanbeige_template;
