//! `deepseek2` — DeepSeek-V2-Lite. The host code that knows this model.
//!
//! The split is the one `docs/arch-split.md` draws: kernels and the shared ops
//! (`ops`, `ffn`, `moe`, `head`, `kv`, `profile`) know shapes only, and the three
//! model-aware layers live here — the load-time plan (`derived`), the layer chain
//! (`forward`), and the attention this architecture uses (`attn`, MLA). The
//! tensor-name table is `names`.

pub mod attn;
pub mod derived;
pub mod forward;
pub mod names;
