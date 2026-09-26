//! Grouped int8 tensor-core GEMM: K-quant weights times q8_1 activations for
//! a batch of token columns, each column routed to one expert of a stack —
//! the prefill shape of a MoE layer, and with a one-expert stack the dense
//! projection. For slots `s < n_slots`,
//! `y[s][n] = Σ_k W[e(s)][n][k] · x[c(s)][k]`, where `e(s)` is the slot's
//! routed expert and `c(s)` its activation column: `s / top_k` when every
//! slot of a token reads the token's input ([`GemmInput::Shared`], gate and
//! up), `s` when each slot has its own ([`GemmInput::PerSlot`], down).
//!
//! Between a gate·up pair and its down, [`GemmKernels::enqueue_swiglu_quant`]
//! writes the down's activations straight from the two GEMMs' rows:
//! `silu(g)·u` per slot column, quantized in the same launch
//! (`gemm_swiglu_quant`), the bytes SwiGLU followed by the quantizer write.
//!
//! Three launches and no host synchronisation between them. The q8_1
//! quantizer every gemv reads ([`Gpu::enqueue_quantize_gemm`] launches the
//! same `q3k_quantize_q8_1` into a [`GemmAct`], which holds more columns
//! than a `Q8Act` may) writes the activations. [`GemmKernels::enqueue_route`]
//! turns the router's ids into a table on the card: the slots grouped by
//! expert, ascending slot order within an expert, cut into tiles of at most
//! [`GEMM_BN`] columns, and the tile count. [`GemmKernels::enqueue_gemm`]
//! launches one block per (tile, `GEMM_BM`-row slab) over a grid sized for
//! the most tiles `n_slots` slots can make; a block past the table's count
//! returns at once, and an expert no slot picked has no tile, so its weights
//! are never read.
//!
//! One file per owner: `grouped.rs` the GEMM (its decode, staging and block
//! walk, and [`GemmKernels::enqueue_gemm`]), `route.rs` the route table,
//! `swiglu.rs` the SwiGLU quantizer's launcher, `act.rs` the activation
//! scratch and its quantizer launch, and `kernels.rs` the one device module
//! every entry is declared in — one `#[cuda_module]`, so one bundle load for
//! the whole family.
//!
//! [`Gpu::enqueue_quantize_gemm`]: crate::Gpu::enqueue_quantize_gemm

mod act;
#[macro_use]
mod grouped;
mod route;
mod swiglu;

// After `grouped`: the entries expand its macros, which are in scope only
// after that module's declaration.
mod kernels;

pub use act::GemmAct;
pub use grouped::{GemmArgs, GemmInput, GemmWeight};
pub use kernels::GemmKernels;
pub use route::{GemmRoute, GemmTile};

/// Slots one route and one GEMM take: a 4096-token ubatch at top-8.
pub const GEMM_MAX_SLOTS: usize = 32_768;
/// Columns (slots) one tile holds: eight n-tiles of eight.
pub const GEMM_BN: usize = 64;
