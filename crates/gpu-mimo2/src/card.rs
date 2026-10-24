//! The routed block's card experts: the common ones
//! ([`bloomery_gpu::card_experts`]) over MiMo's stacks and counts.
//!
//! MiMo facts that fix this file: the routed block has no shared expert, so
//! the shadow is the card slots alone and the layer's back adds the host's sum
//! to `acc + x1`; the routed SwiGLU is the description's limit (0: plain); no
//! expert tier hangs on these experts.

use bloomery_gpu::card_experts::{CardExperts, CardFamily};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::router::mimo2::{N_EXPERT, N_USED};
use model::arch::mimo2::names;
use models::LayerSpec;

/// A layer's routed gate, up and down stacks.
fn stacks(l: usize) -> [String; 3] {
    [
        names::ffn_gate_exps(l),
        names::ffn_up_exps(l),
        names::ffn_down_exps(l),
    ]
}

/// MiMo's card experts: 256 routed experts, 8 a token.
const FAMILY: CardFamily = CardFamily {
    what: "mimo2 CardExperts",
    program: model::placement::accepts::Program::Mimo2Body,
    n_expert: N_EXPERT,
    n_used: N_USED,
    stacks,
};

/// The card experts of `layers` (the description) as `map` places them and
/// `w` holds them, `n` wide with routed experts `ff` wide, with buffers for a
/// pass of `rows` rows when `rows` is more than one. Load-time only.
pub(crate) fn experts(
    gpu: &Gpu,
    w: &Weights,
    layers: &[LayerSpec],
    map: &SlotMap,
    n: usize,
    ff: usize,
    rows: usize,
) -> Result<CardExperts, GpuError> {
    CardExperts::new(gpu, w, FAMILY, layers, map, (n, ff), rows)
}
