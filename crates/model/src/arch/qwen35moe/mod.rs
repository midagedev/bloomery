//! `qwen35moe` — Qwen3.5/3.6 mixture-of-experts (Qwen3.6-35B-A3B), and its
//! variant `qwen4exp` (Qwen3.8-Flash-Next). The file's keys (`hparams`),
//! every tensor's role (`roles`) and the typed description (`spec`), which
//! the coverage check lists what a program has to run from; for qwen4exp the
//! tensor names its program reads (`names`), the plan from the headers
//! (`place`) and the host tier's view of a routed layer (`host`).
//!
//! The trunk interleaves gated delta-rule (GDN) layers with gated GQA layers;
//! every layer routes to experts and runs a sigmoid-gated shared expert.
//! qwen4exp wraps every layer in gated-residual hyper-connections, selects an
//! attention layer's positions by a mean-pool top-k, and adds a per-layer
//! n-gram embedding (PLE) on one GDN layer. A qwen4exp MTP draft file is read
//! against its target by `mtp`. The line numbers cited are
//! llama.cpp's `src/models/qwen35moe.cpp`, `src/models/qwen4exp.cpp` and
//! `src/llama-hparams.cpp` unless another file is named.

pub mod host;
pub mod hparams;
pub mod mtp;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
