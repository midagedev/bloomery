//! The qwen3moe family's half of the GPU engine: two chain bodies over one
//! layer body (`dispatch::layer`) — [`Body`], Qwen3-30B-A3B (every layer
//! QK-normed grouped-query attention over the layer's own K/V planes, then
//! eight routed experts; no shared expert, no dense layer), and [`Body35`],
//! Qwen3.6-35B-A3B (gated-delta-rule layers between gated GQA layers at head
//! 256, each layer's experts carrying a sigmoid-gated shared expert) — each
//! layer's kind a load-time plan (`plan`); [`Body38`], Qwen3.8-Flash-Next,
//! over its own plan and program (`plan38`, [`program38`]: the layers in
//! gated-residual streams, a selecting attention, every routed expert on the
//! host tier); and the kernels whose constants
//! are this family's and nothing else's — the attention projections, the
//! routers, the experts' gate·up and combine, and the head's projection with
//! its argmax. The shape kernels it runs over — the Q4_K embedding row, the
//! NEOX norm/rope/append, the grouped-query flash, the delta rule, the Q6_K
//! down `_sel` — live in the crate root beside the runtime.

mod body;
mod body35;
mod body38;
mod delta;
mod dispatch;
pub mod experts;
pub mod head_argmax;
mod image;
mod plan;
mod plan38;
mod prefill;
mod program;
pub mod program38;
pub mod proj;
pub mod router;
mod scratch;
mod scratch38;
mod taps;
mod taps35;
pub mod ubatch;
mod wide;

pub use body::{Body, DecodeInput, FlashKind, OpenOpts};
pub use body35::{Body35, DecodeInput35, LayerKind35, Open35};
pub use body38::{
    ALLOWED, Body38, DecodeInput38, LayerKind38, Prompt38, Qwen38Model, RouteTap, Store38Host,
};
pub use taps35::{Delta35Run, Ffn35Run, Gqa35Run, Layer35Run, Mixer35Run, StoreHost};

/// Qwen3.6-35B-A3B on one card.
pub type Qwen35moeModel = crate::GpuModel<Body35>;
pub use prefill::{PrefillPath, PrefillPlan, PrefillStep};
pub use taps::LayerRun;
