//! `qwen35moe` — Qwen3.5/3.6 mixture-of-experts (Qwen3.6-35B-A3B). A header
//! reader only: no program here runs this architecture yet. The file's keys
//! (`hparams`), every tensor's role (`roles`) and the typed description
//! (`spec`) exist so the coverage check can list what a program has to run.
//!
//! The trunk interleaves gated delta-rule (GDN) layers with gated GQA layers;
//! every layer routes to experts and runs a sigmoid-gated shared expert. The
//! line numbers cited are llama.cpp's `src/models/qwen35moe.cpp` and
//! `src/llama-hparams.cpp` unless another file is named.

pub mod hparams;
pub mod roles;
pub mod spec;
