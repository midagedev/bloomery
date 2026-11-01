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
//! `swiglu.rs` the SwiGLU quantizer's launcher, `swiglu32.rs` the SwiGLU
//! epilogue an up GEMM of either family writes the down's q8 blocks of 32
//! through, `act.rs` the activation
//! scratch and its quantizer launch, and `kernels.rs` the one device module
//! every entry is declared in — one `#[cuda_module]`, so one bundle load for
//! the whole family.
//!
//! The 32-value family is the same shape for the weights whose scales come
//! per 32 values (Q8_0, IQ4_NL, Q5_1), over [`GemmAct32`] activations (one scale and
//! one code sum per 32 values, K any multiple of 32), in a device module of
//! its own ([`Gemm32Kernels`], `kernels32.rs`): `gemm32.rs` the GEMM and its
//! numeric contract, `act32.rs` the activations and their two quantizers,
//! `remap.rs` the route table over ids mapped to a card stack's slots or to
//! the host, and `f32tile.rs` the wide F32 product. Its GEMMs read the same
//! [`GemmRoute`] as the K-quant ones.
//!
//! `batch.rs` is the sequence over both families a remapped card route runs
//! between the model's places launch and its card sum
//! ([`enqueue_gemm_batch`]): the route table, the gate and up GEMMs, the
//! SwiGLU of the card slots' places and the down GEMM.
//!
//! [`Gpu::enqueue_quantize_gemm`]: crate::Gpu::enqueue_quantize_gemm

mod act;
#[macro_use]
mod grouped;
mod route;
mod swiglu;
#[macro_use]
mod swiglu32;

// After `grouped`: the entries expand its macros, which are in scope only
// after that module's declaration.
mod kernels;

mod act32;
mod f32tile;
#[cfg(bloomery_nvfp4_mma)]
mod nvfp4mma;
#[macro_use]
mod gemm32;
mod remap;

// After `gemm32`, for the same reason.
mod kernels32;

mod batch;

pub use act::GemmAct;
pub(crate) use act32::quant32_group;
pub use act32::{GemmAct32, MmaForm};
pub use batch::{DownW, DownWin, GateUp, enqueue_gemm_batch};
pub use f32tile::LINE as F32_TILE_LINE;
pub(crate) use f32tile::mxfp8_w8a8_reads;
pub(crate) use f32tile::{
    Bf16Cols, Bf16Out, Bf16Rows, Bf16WordCols, F32Cols, F32Rows, F32T_THREADS_U32, HcUpMixArgs,
    MMA_COLS, MMA_ROWS, MMA_THREADS, MMA_WARP_COLS, MMA_WARP_ROWS, MMA_WARP_SCRATCH, MmaCols,
    MmaRows, MmaTile, NARROW_ROWS, NARROW_WORDS, OutForm, STAGE_FLOATS, STAGES, TileRows,
    bf16_mma_body, bf16_mma_launch, bf16_mma_narrow, bf16_mma_tile, f32_tile_body, form_col,
    land_values, mma_launch, mma_launch_words, mma_tile_origin, store_refused, tile_launch,
};
pub use f32tile::{Mxfp8NormGateArgs, ProductOut};
pub use gemm32::{GateUp32Args, Gemm32Args, Gemm32Weight};
pub use grouped::{GemmArgs, GemmInput, GemmWeight};
pub use kernels::GemmKernels;
pub use kernels32::Gemm32Kernels;
pub use route::{GemmRoute, GemmTile};
pub use swiglu32::SwigluSlots;

/// Slots one route and one GEMM take: a 4096-token ubatch at ten slots a
/// token (ten routed experts, or nine and the shared one).
pub const GEMM_MAX_SLOTS: usize = 40_960;
/// Columns (slots) one tile holds: eight n-tiles of eight.
pub const GEMM_BN: usize = 64;
/// Values one step of the 32-value family's K walk covers: two blocks. A
/// [`GemmAct32`] column is padded to whole steps.
pub const GEMM32_STEP: usize = 64;
