//! `qwen3moe` — Qwen3 mixture-of-experts (Qwen3-30B-A3B, Qwen3-235B-A22B). The
//! host code that knows this model: its hyperparameters (`hparams`), the
//! tensor names the step reads (`names`), the role of every tensor (`roles`),
//! the KV bytes each layer holds (`kv`), the path from a file to its
//! placement plan (`place`) and the host tier's layer spec (`host`).

pub mod host;
pub mod hparams;
pub mod kv;
pub mod names;
pub mod place;
pub mod roles;
