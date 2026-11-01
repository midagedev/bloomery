//! `llama` — the dense Llama architecture (grouped-query attention on the
//! pairwise rope, a SwiGLU feed-forward block, a head of its own or tied to
//! the token embedding), and `qwen3`, the dense Qwen3 trunk that differs only
//! by its architecture facts: per-head q/k norms and the NEOX rope. The host
//! code that knows them: the hyperparameters (`hparams`, the rope pairing the
//! one family fact), the role of every tensor (`roles`) and the typed model
//! description (`spec`). The tensor names are the ones every family spells
//! alike ([`crate::arch::names`]).

pub mod hparams;
pub mod roles;
pub mod spec;

#[cfg(test)]
mod nanbeige_template;
