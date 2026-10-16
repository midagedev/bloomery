//! `nemotron_h_moe`: Nemotron-H with latent mixture-of-experts layers in
//! place of the dense MLP ones — Nemotron 3.5 Bolt 74B-A9B, whose MoE layers
//! each carry their own expert count and width. The file's keys
//! (`hparams`), every tensor's role (`roles`), the tensor names (`names`)
//! and the typed description (`spec`). The Mamba-2 and attention layers are
//! read by the hybrid trunk's common pieces (`arch::hybrid`), which
//! `nemotron_h` reads its own through, and the whole-card plan is the trunk's
//! too (`place`). `Program::NemotronHMoeBody` runs it
//! (`crates/gpu/src/arch/nemotron_h_moe`). The dialect's authority is
//! llama.cpp's `src/models/nemotron-h.cpp`, which loads both architectures.

pub mod fixture;
pub mod hparams;
pub mod host;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
