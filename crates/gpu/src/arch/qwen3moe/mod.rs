//! The qwen3moe family's half of the GPU engine: two chain bodies over one
//! layer body (`dispatch::layer`) — [`Body`], Qwen3-30B-A3B (every layer
//! QK-normed grouped-query attention over the layer's own K/V planes, then
//! eight routed experts; no shared expert, no dense layer), and [`Body35`],
//! Qwen3.6-35B-A3B (gated-delta-rule layers between gated GQA layers at head
//! 256, each layer's experts carrying a sigmoid-gated shared expert) — each
//! layer's kind a load-time plan (`plan`); and the kernels whose constants
//! are this family's and nothing else's — the attention projections, the
//! routers, the experts' gate·up and combine, and the head's projection with
//! its argmax. The shape kernels it runs over — the Q4_K embedding row, the
//! NEOX norm/rope/append, the grouped-query flash, the delta rule, the Q6_K
//! down `_sel` — live in the crate root beside the runtime.

mod body;
mod body35;
mod delta;
mod dispatch;
pub mod experts;
pub mod head_argmax;
mod plan;
mod prefill;
mod program;
pub mod proj;
pub mod router;
mod scratch;
mod taps;
mod taps35;
pub mod ubatch;

pub use body::{Body, DecodeInput, FlashKind, OpenOpts};
pub use body35::{Body35, DecodeInput35, LayerKind35};
pub use taps35::{Delta35Run, Ffn35Run, Gqa35Run, Layer35Run, Mixer35Run, StoreHost};

/// Qwen3.6-35B-A3B on one card.
pub type Qwen35moeModel = crate::GpuModel<Body35>;
pub use prefill::{PrefillPath, PrefillPlan, PrefillStep};
pub use taps::LayerRun;
