//! The family's own layer: a latent mixture of routed experts with a shared expert, every weight's
//! name and count the layer's own (the file's MoE layers differ in expert count and width).

use crate::hybrid::plan::LayerPlan;

/// A latent MoE layer: the pre-norm, the router with its selection bias, the latent pair around
/// the kept experts, the routed stacks and the shared expert — every part by its file name, the
/// types and forms the family resolves at load.
#[derive(Clone, Debug, PartialEq)]
pub struct MoePlan {
    pub norm: String,
    /// The F32 router rows, `n_expert × hidden` (`ffn_gate_inp`).
    pub router: String,
    /// The selection bias, `n_expert` values (`exp_probs_b`).
    pub bias: String,
    /// The layer's expert count: its router's rows, which its stacks' last dim and its bias
    /// repeat.
    pub n_expert: usize,
    /// One expert's values (`expert_feed_forward_length` at the layer).
    pub ff: usize,
    /// The experts a token keeps (`expert_used_count` at the layer).
    pub top_k: usize,
    /// `ffn_latent_down` (`hidden → latent`), where the file has a latent.
    pub latent_down: Option<String>,
    /// `ffn_latent_up` (`latent → hidden`), where the file has a latent.
    pub latent_up: Option<String>,
    /// The routed up stack, every expert's up rows in the file's blocks (`ffn_up_exps`).
    pub up_exps: String,
    /// The routed down stack, every expert's down rows (`ffn_down_exps`).
    pub down_exps: String,
    /// The shared expert's up (`ffn_up_shexp`).
    pub shared_up: String,
    /// The shared expert's down (`ffn_down_shexp`).
    pub shared_down: String,
}

/// One layer of a nemotron_h_moe chain.
pub type Plan = LayerPlan<MoePlan>;
