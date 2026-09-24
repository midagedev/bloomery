//! `deepseek2` — DeepSeek-V2-Lite. The host code that knows this model.
//!
//! The split is the one `docs/arch-split.md` draws: kernels and the shared ops
//! (`ops`, `ffn`, `moe`, `head`, `kv`, `profile`) know shapes only, and the
//! model-aware layers live here — the hyperparameters and the values this chain
//! refuses (`hparams`), tensor resolution (`plan`) over the name table
//! (`names`), the load-time weight state (`derived`), the layer chain
//! (`forward`), and the attention this architecture uses (`attn`, MLA).

pub mod attn;
pub mod derived;
pub mod forward;
pub mod hparams;
pub mod names;
pub mod plan;
