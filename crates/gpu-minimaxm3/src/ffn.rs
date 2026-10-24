//! The feed-forward sub-layer: the common routed block with its shared expert
//! ([`bloomery_gpu_deepseek41::shared_routed`]) over MiniMax's router
//! `sigmoid_128_4`, and the dense block of the leading layers the same module
//! runs ([`dense`]).
//!
//! MiniMax facts that fix this file: 128 routed experts, the top 4; every
//! block, dense, routed and shared, runs SwiGLU-OAI; the shared expert has no
//! sigmoid gate; the first layers of the file are dense blocks of their own
//! width, and the rest routed with one shared expert.

use bloomery_gpu::card_experts::CardFamily;
use bloomery_gpu_deepseek41::router::sigmoid_128_4::{N_EXPERT, N_USED, RouterKernels};
use bloomery_gpu_deepseek41::shared_routed;
use model::arch::names;

pub use bloomery_gpu_deepseek41::shared_routed::{
    GATED_FROM, RoutedDims, back, dense, front, shadow,
};

/// The card experts' family: the file's stack names and the router's counts.
pub const CARD: CardFamily = CardFamily {
    what: "minimaxm3 CardExperts",
    program: model::placement::accepts::Program::MiniMaxM3Body,
    n_expert: N_EXPERT,
    n_used: N_USED,
    stacks: names::routed_stacks,
};

/// MiniMax's routed blocks.
pub type Routed = shared_routed::Routed<RouterKernels>;
