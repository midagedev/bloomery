//! `qwen3moe` — Qwen3 mixture-of-experts (Qwen3-30B-A3B, Qwen3-235B-A22B). The
//! host code that knows this model: its hyperparameters (`hparams`), the
//! tensor names the step reads (`names`), the role of every tensor
//! (`roles`), and the plans of the program's placed loads (`place`).

pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
