//! `mimo2` — MiMo-V2.6-Flash. The file's keys (`hparams`), every tensor's
//! role (`roles`), the typed description (`spec`), the tensor names the
//! program reads (`names`), the host tier's view of a routed layer (`host`),
//! the file planned onto a machine (`place`) and the facts of the program that
//! runs it (`program`); the body that runs the file is `crates/gpu-mimo2`.
//!
//! Every block is a GQA layer over two head widths — a key head wider than
//! the value head — whose KV-head count, window kind and rope base are the
//! layer's own, read from the per-layer arrays. Layer 0 and the next-token
//! blocks run a dense SwiGLU block; the rest route by a sigmoid noaux_tc
//! router over MXFP4 expert stacks. The dialect's authority is llama.cpp's
//! `src/models/mimo2.cpp`.

pub mod host;
pub mod hparams;
pub mod names;
pub mod place;
pub mod program;
pub mod roles;
pub mod spec;
