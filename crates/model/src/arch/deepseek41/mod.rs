//! `deepseek41` — DeepSeek-V4.1-Flash. The host code that knows this model: its
//! hyperparameters and per-layer kinds (`hparams`), the tensor names the step
//! reads (`names`), the role of every tensor (`roles`), the KV bytes each layer
//! holds (`kv`), the path from a file to its placement plan (`place`), the
//! host tier's layer spec (`host`) and the integers each step's graph reads
//! (`plan`).

pub mod host;
pub mod hparams;
pub mod kv;
pub mod names;
pub mod place;
pub mod plan;
pub mod roles;
