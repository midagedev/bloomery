//! `nemotron_h`: a dense hybrid of Mamba-2, squared-ReLU MLP and no-position
//! attention. The file's keys (`hparams`), every tensor's role (`roles`),
//! the tensor names (`names`) and the typed description (`spec`). No program
//! runs it. The dialect's authority is llama.cpp's `src/models/nemotron-h.cpp`.

pub mod hparams;
pub mod names;
pub mod roles;
pub mod spec;
