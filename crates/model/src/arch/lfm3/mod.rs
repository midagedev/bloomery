//! `lfm3`: gated short-conv layers and grouped-query attention layers (sliding window or full) in
//! one trunk, a dense SwiGLU block in every layer, sandwich norms around both sublayers and a
//! head tied to the token embedding. The file's keys (`hparams`), every tensor's role (`roles`),
//! the tensor names (`names`), the typed description (`spec`), the placement inputs (`place`) and the gate fixture's spec
//! (`fixture`).
//! No program runs the description yet: the coverage check lists each part of it by name.
//! The dialect's authority is llama.cpp's `src/models/lfm3.cpp`.

pub mod fixture;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
