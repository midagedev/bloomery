//! bloomery-gpu-deepseek41 — the DeepSeek-V4.1 device code: the
//! `#[cuda_module]`s only this architecture runs. They live outside
//! `bloomery-gpu` so that a compiler defect in one of them breaks the builds
//! that depend on this crate and no other: a build without this crate in its
//! graph never compiles V4.1 device code. Device bodies shared with V2-Lite
//! are called as `bloomery_gpu::cores::*`; a body is reachable from here when
//! it is `pub` and `#[inline(always)]`, and it inlines into the same
//! instructions it compiles to in its own crate.
//!
//! Every kernel entry in this crate is named `ds41_<op>`: cuda-oxide derives
//! a kernel's host symbol from the entry name alone, not from its crate or
//! module, so an entry name another crate also declares fails to link.
//!
//! One module per V4.1-only op, all declared here up front, so writing an
//! op's kernels touches its own module and not this file.
//!
//! Like `bloomery-gpu`, this crate is built with `cargo oxide` only.

pub mod attn;
pub mod body;
pub mod chain;
pub mod compress;
pub mod engram_gate;
pub mod experts;
pub mod hc;
pub mod hc_f32;
pub mod index_key;
pub mod indexer;
pub mod markov;
pub mod params;
pub mod rope;
pub mod router;
