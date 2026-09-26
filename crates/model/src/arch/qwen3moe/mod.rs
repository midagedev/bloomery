//! `qwen3moe` — Qwen3 mixture-of-experts (Qwen3-30B-A3B, Qwen3-235B-A22B). The
//! host code that knows this model: its hyperparameters (`hparams`), the
//! tensor names the step reads (`names`) and the role of every tensor
//! (`roles`).

pub mod hparams;
pub mod names;
pub mod roles;
