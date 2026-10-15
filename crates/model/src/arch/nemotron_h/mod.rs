//! `nemotron_h`: a dense hybrid of Mamba-2, squared-ReLU MLP and no-position
//! attention. The file's keys (`hparams`), every tensor's role (`roles`),
//! the tensor names (`names`), the typed description (`spec`) and the gate
//! fixture's spec (`fixture`). The card body is `gpu::arch::nemotron_h`.
//! The dialect's authority is llama.cpp's `src/models/nemotron-h.cpp`.

pub mod fixture;
pub mod hparams;
pub mod names;
pub mod roles;
pub mod spec;
