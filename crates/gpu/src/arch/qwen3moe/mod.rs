//! The qwen3moe (Qwen3-30B-A3B) half of the GPU engine: the chain body
//! [`Body`] (one layer is QK-normed grouped-query attention over the layer's
//! own K/V planes, then eight routed experts; no shared expert, no dense
//! layer), and the kernels whose constants are this architecture's and
//! nothing else's — the attention projections, the router, the experts'
//! gate·up and combine, and the head's projection with its argmax. The
//! shape kernels it runs over — the Q4_K embedding row, the NEOX
//! norm/rope/append, the grouped-query flash, the Q6_K down `_sel` — live in
//! the crate root beside the runtime.

mod body;
mod dispatch;
pub mod experts;
pub mod head_argmax;
mod prefill;
pub mod proj;
pub mod router;
mod scratch;
mod taps;
pub mod ubatch;

pub use body::{Body, DecodeInput};
pub use prefill::{PrefillPath, PrefillPlan, PrefillStep};
pub use taps::LayerRun;
