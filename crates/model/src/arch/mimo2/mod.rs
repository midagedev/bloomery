//! `mimo2` — MiMo-V2.6-Flash. The file's keys (`hparams`), every tensor's
//! role (`roles`), the typed description (`spec`), and the tensor names the
//! program reads (`names`); the program is a later round's.
//!
//! Every block is a GQA layer over two head widths — a key head wider than
//! the value head — whose KV-head count, window kind and rope base are the
//! layer's own, read from the per-layer arrays. Layer 0 and the next-token
//! blocks run a dense SwiGLU block; the rest route by a sigmoid noaux_tc
//! router over MXFP4 expert stacks. The dialect's authority is llama.cpp's
//! `src/models/mimo2.cpp`.

pub mod hparams;
pub mod names;
pub mod roles;
pub mod spec;
