//! The weight names of one deepseek2 layer — `blk.{l}.<stem>.weight` and the
//! derived plane's name, the one table in this architecture's chain that is a
//! string. Built at load so the step never formats one (docs/gpu-design.md
//! decision 4).

use crate::weights::Weights;

/// The weight names one layer's chain looks up, built once at load so the
/// step never formats a string (decision 4). `routed` is read from the
/// file, not from the layer index: a layer is MoE exactly when its router
/// weight is resident.
pub(super) struct LayerNames {
    pub(super) layer: usize,
    pub(super) attn_norm: String,
    pub(super) attn_q: String,
    pub(super) attn_kv_a_mqa: String,
    pub(super) attn_kv_a_norm: String,
    pub(super) attn_kv_b: String,
    pub(super) attn_output: String,
    pub(super) ffn_norm: String,
    pub(super) ffn_gate: String,
    pub(super) ffn_up: String,
    pub(super) ffn_down: String,
    pub(super) ffn_gate_inp: String,
    pub(super) ffn_gate_exps: String,
    pub(super) ffn_up_exps: String,
    pub(super) ffn_down_exps: String,
    pub(super) ffn_gate_shexp: String,
    pub(super) ffn_up_shexp: String,
    pub(super) ffn_down_shexp: String,
    /// The derived q_nope2 planes' name — a `format!` of the layer index, so
    /// it is resolved here and never on the step's path (decision 4).
    pub(super) derived: String,
    pub(super) routed: bool,
}

impl LayerNames {
    /// Every name of block `l`, and whether that block routes.
    pub(super) fn new(w: &Weights, l: usize) -> LayerNames {
        let n = |stem: &str| format!("blk.{l}.{stem}.weight");
        let ffn_gate_inp = n("ffn_gate_inp");
        LayerNames {
            layer: l,
            routed: w.get(&ffn_gate_inp).is_some(),
            attn_norm: n("attn_norm"),
            attn_q: n("attn_q"),
            attn_kv_a_mqa: n("attn_kv_a_mqa"),
            attn_kv_a_norm: n("attn_kv_a_norm"),
            attn_kv_b: n("attn_kv_b"),
            attn_output: n("attn_output"),
            ffn_norm: n("ffn_norm"),
            ffn_gate: n("ffn_gate"),
            ffn_up: n("ffn_up"),
            ffn_down: n("ffn_down"),
            ffn_gate_inp,
            ffn_gate_exps: n("ffn_gate_exps"),
            ffn_up_exps: n("ffn_up_exps"),
            ffn_down_exps: n("ffn_down_exps"),
            ffn_gate_shexp: n("ffn_gate_shexp"),
            ffn_up_shexp: n("ffn_up_shexp"),
            ffn_down_shexp: n("ffn_down_shexp"),
            derived: crate::weights::derived_name(l),
        }
    }
}
