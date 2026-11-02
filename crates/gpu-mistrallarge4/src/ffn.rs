//! The routed block with its shared expert: the common one
//! ([`bloomery_gpu_deepseek41::shared_routed`]) over the softmax router entry
//! ([`SoftmaxEntry`]) and the file's stacks.
//!
//! Mistral Large 4 facts that fix this file: 128 routed experts, the top 4 by
//! softmax probability with the kept four renormalized; one ungated shared
//! expert beside them; the routed sum joins the shared output before the
//! residual (mainline's `(ffn_moe_out + ffn_shexp) + ffn_inp`,
//! [`ResidualOrder::RoutedSharedResidual`]). A slot map that names an expert
//! tier card makes the common block run the layers the tier holds experts of
//! as tier layers (the join sums the stage card's and the tier's slots in
//! slot order), with nothing more from this file: the tier's own pieces are
//! `crate::tier`'s.

use bloomery_gpu::card_experts::CardFamily;
use bloomery_gpu::card_leg::ExpertInput;
use bloomery_gpu::route_softmax::SoftmaxEntry;
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu_deepseek41::shared_routed::{self, ResidualOrder};
use model::arch::names;
use model::placement::accepts::Program;

pub use bloomery_gpu_deepseek41::shared_routed::{
    FfnConfig, JoinTerms, RoutedDims, back, dense, front, shadow,
};

/// The experts a token routes to.
pub const N_USED: usize = 4;

/// The router: the softmax entry over its 128 experts, the top [`N_USED`].
pub type Router = SoftmaxEntry<N_USED>;

/// The experts a layer routes over: the router's own width.
pub const N_EXPERT: usize = <Router as RouterEntry>::N_EXPERT;

/// The card experts' family: the file's stack names and the router's counts.
pub const CARD: CardFamily = CardFamily {
    what: "mistrallarge4 CardExperts",
    program: Program::MistralLarge4Body,
    n_expert: N_EXPERT,
    n_used: N_USED,
    stacks: names::routed_stacks,
};

/// The block's arithmetic: the routed sum and the shared output joined before
/// the residual, the experts reading the normed row as it is.
pub const CONFIG: FfnConfig = FfnConfig {
    residual: ResidualOrder::RoutedSharedResidual,
    expert_input: ExpertInput::Direct,
};

/// Mistral Large 4's routed blocks.
pub type Routed = shared_routed::Routed<Router>;
