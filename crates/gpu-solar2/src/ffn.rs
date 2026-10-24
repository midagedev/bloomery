//! The routed block with its shared expert: the common one
//! ([`bloomery_gpu_deepseek41::shared_routed`]) over Solar's router
//! `sigmoid_320_8` and its stacks.
//!
//! Solar facts that fix this file: 320 routed experts, the top 8; the routed
//! experts' SwiGLU and the shared expert's are the description's limits (0:
//! plain); the shared expert has no sigmoid gate and is 1280 wide, its down
//! Q6_K or Q4_K by layer.

use bloomery_gpu::card_experts::CardFamily;
use bloomery_gpu_deepseek41::router::sigmoid_320_8::{N_EXPERT, N_USED, RouterKernels};
use bloomery_gpu_deepseek41::shared_routed;
use model::arch::names;

pub use bloomery_gpu_deepseek41::shared_routed::{GATED_FROM, RoutedDims, back, front, shadow};

/// The card experts' family: the file's stack names and the router's counts.
pub const CARD: CardFamily = CardFamily {
    what: "solar2 CardExperts",
    program: model::placement::accepts::Program::Solar2Body,
    n_expert: N_EXPERT,
    n_used: N_USED,
    stacks: names::routed_stacks,
};

/// Solar's routed blocks.
pub type Routed = shared_routed::Routed<RouterKernels>;
