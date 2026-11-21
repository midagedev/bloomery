//! The role of every tensor in an Inkling 2 file, by exact name: the model-level names (the
//! embedding's norm and the final mix's query among them, the loop's message queries under the
//! unsloth dialect), then the per-layer stems by the groups a layer's kind carries
//! ([`stems_of`]): the attention block every layer holds, the windowed layer's shared input norm
//! or the windowless layer's three input norms and input-dependent conv taps, the block
//! attention residuals' queries, the routed block's fused stacks, and the core layers' message
//! controllers in the file's dialect's spelling. A name the layer's kind does not hold — the
//! other dialect's loop stems, a controller on a non-core layer, a fused projection, a block
//! past the stored layers — fails the file with every such name listed.
//!
//! The loop's controllers and message queries sit with the attention rows: every token of every
//! pass reads them on the layer's card, serial with the attention.

use gguf::Split;

use crate::arch::classify::{Counts, classify_with, role_by_layer};
use crate::arch::stems::{Stem, join, opt, req};
use crate::placement::{ModelTensors, PlacementError, Role};

use super::hparams::{Dialect, Hparams};

/// What a layer is: whether it is windowed, whether it is a core layer of the loop, and whether
/// it is the trunk's first layer (which carries no attention mix query: with nothing banked
/// there is nothing to mix).
pub(super) type LayerKind = (bool, bool, bool);

/// The attention block every layer carries: the projections, the bias table and its clips, the
/// sink projection, the k and v conv taps, and the attention output's conv.
const ATTN: &[Stem] = &[
    req("attn_q.weight", Role::Attention),
    req("attn_k.weight", Role::Attention),
    req("attn_v.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
    req("attn_gate.weight", Role::Attention),
    req("attn_q_clip.weight", Role::Attention),
    req("attn_sink_proj.weight", Role::Attention),
    req("attn_rel.weight", Role::Attention),
    req("attn_rel_bias.weight", Role::Attention),
    req("attn_rel_clip.weight", Role::Attention),
    req("attn_k_conv.weight", Role::Attention),
    req("attn_v_conv.weight", Role::Attention),
    req("attn_post_conv.weight", Role::Attention),
];

/// A windowed layer's shared input norm.
const LOCAL: &[Stem] = &[req("attn_norm.weight", Role::Attention)];

/// A windowless layer's own input norms and input-dependent k and v conv taps.
const GLOBAL: &[Stem] = &[
    req("attn_q_inp_norm.weight", Role::Attention),
    req("attn_k_inp_norm.weight", Role::Attention),
    req("attn_v_inp_norm.weight", Role::Attention),
    req("attn_k_conv_proj.weight", Role::Attention),
    req("attn_v_conv_proj.weight", Role::Attention),
];

/// The feed-forward block's conv and mix query, the pre-FFN norm, the router with its optional
/// selection bias, and the fused routed stacks.
const FFN: &[Stem] = &[
    req("ffn_post_conv.weight", Role::FfnNorm),
    req("ffn_res_query.weight", Role::FfnNorm),
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate_inp.weight", Role::Router),
    opt("exp_probs_b.bias", Role::Router),
    req("ffn_gate_up_exps.weight", Role::RoutedExperts),
    req("ffn_down_exps.weight", Role::RoutedExperts),
];

/// The unsloth dialect's message controllers: `down`, `gain` and `forget` for each of the four
/// modules (attention, k, v, ffn).
pub(super) const LOOP_UNSLOTH: &[Stem] = &join::<12>(&[
    &[
        req("loop_attn_down.weight", Role::Attention),
        req("loop_attn_gain.weight", Role::Attention),
        req("loop_attn_forget.weight", Role::Attention),
    ],
    &[
        req("loop_k_down.weight", Role::Attention),
        req("loop_k_gain.weight", Role::Attention),
        req("loop_k_forget.weight", Role::Attention),
    ],
    &[
        req("loop_v_down.weight", Role::Attention),
        req("loop_v_gain.weight", Role::Attention),
        req("loop_v_forget.weight", Role::Attention),
    ],
    &[
        req("loop_mlp_down.weight", Role::Attention),
        req("loop_mlp_gain.weight", Role::Attention),
        req("loop_mlp_forget.weight", Role::Attention),
    ],
]);

/// llama.cpp master's dialect's message queries and controllers.
pub(super) const LOOP_MASTER: &[Stem] = &join::<13>(&[
    &[req("loop_msg_query.weight", Role::Attention)],
    &[
        req("loop_q_ctrl_down.weight", Role::Attention),
        req("loop_q_ctrl_gain.weight", Role::Attention),
        req("loop_q_ctrl_forget.weight", Role::Attention),
    ],
    &[
        req("loop_k_ctrl_down.weight", Role::Attention),
        req("loop_k_ctrl_gain.weight", Role::Attention),
        req("loop_k_ctrl_forget.weight", Role::Attention),
    ],
    &[
        req("loop_v_ctrl_down.weight", Role::Attention),
        req("loop_v_ctrl_gain.weight", Role::Attention),
        req("loop_v_ctrl_forget.weight", Role::Attention),
    ],
    &[
        req("loop_ffn_ctrl_down.weight", Role::Attention),
        req("loop_ffn_ctrl_gain.weight", Role::Attention),
        req("loop_ffn_ctrl_forget.weight", Role::Attention),
    ],
]);

/// Every kind of layer a file may hold: windowed or windowless, a core layer of the loop or not,
/// the trunk's first layer or another.
pub(super) const KINDS: [LayerKind; 8] = [
    (true, false, true),
    (true, false, false),
    (true, true, true),
    (true, true, false),
    (false, false, true),
    (false, false, false),
    (false, true, true),
    (false, true, false),
];

/// The attention's mix query, every layer but the trunk's first: `attn_res_query`, one row
/// windowed and three windowless.
const MIX: &[Stem] = &[req("attn_res_query.weight", Role::Attention)];

/// The stem groups a layer of `kind` carries, in the file's `dialect`.
pub(super) fn stems_of(
    dialect: Dialect,
    (windowed, core, first): LayerKind,
) -> Vec<&'static [Stem]> {
    let mut groups: Vec<&[Stem]> = vec![ATTN, if windowed { LOCAL } else { GLOBAL }];
    if !first {
        groups.push(MIX);
    }
    groups.push(FFN);
    if core {
        groups.push(match dialect {
            Dialect::Unsloth => LOOP_UNSLOTH,
            Dialect::Master => LOOP_MASTER,
        });
    }
    groups
}

/// Every model-level tensor's role by exact name, the loop's message queries under the unsloth
/// dialect alone — a layer each, the core's first: the plan reads them with that layer's
/// attention rows, and a model-level attention tensor has no stage of its own. `None` for a
/// per-layer name (the classifier's walk takes it from there).
pub(super) fn model_level(
    name: &str,
    dialect: Dialect,
    core_start: usize,
) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd_norm.weight" => Some((Role::TokenEmbedding, None)),
        "output_res_query.weight" => Some((Role::Head, None)),
        "loop_res_query.weight" | "loop_kv_res_query.weight" if dialect == Dialect::Unsloth => {
            Some((Role::Attention, Some(core_start)))
        }
        _ => None,
    }
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let core_start = hp.loop_.map_or(0, |loop_| loop_.core_start);
    classify_with(
        split,
        |name| {
            model_level(name, hp.dialect, core_start)
                .or_else(|| role_by_layer(name, hp.n_layer, |l| stems_of(hp.dialect, hp.kind(l))))
        },
        Counts::new(hp.n_layer, hp.n_expert, hp.n_used[0]),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ATTN, FFN, GLOBAL, KINDS, LOCAL, LOOP_MASTER, LOOP_UNSLOTH, model_level, stems_of,
    };
    use crate::arch::classify::{assert_one_role_per_stem, role_by_layer};
    use crate::placement::Role;

    /// A tensor's role and layer by its name in a 6-layer file whose layer 2 is windowless, the
    /// core [2, 4) of a loop, under the dialect `dialect`.
    fn role_of(name: &str, dialect: super::Dialect) -> Option<(Role, Option<usize>)> {
        super::model_level(name, dialect, 2).or_else(|| {
            role_by_layer(name, 6, |l| {
                stems_of(dialect, (l != 2, (2..4).contains(&l), l == 0))
            })
        })
    }

    /// No stem is listed twice within a layer kind's groups, a stem two groups share names one
    /// role in both, and the group counts are the table's.
    #[test]
    fn the_tables_list_each_stem_once() {
        for dialect in [super::Dialect::Unsloth, super::Dialect::Master] {
            let groups: Vec<_> = KINDS
                .into_iter()
                .flat_map(|k| stems_of(dialect, k))
                .collect();
            assert_one_role_per_stem(&groups);
        }
        assert_eq!(
            (ATTN.len(), LOCAL.len(), GLOBAL.len(), FFN.len()),
            (13, 1, 5, 7)
        );
        assert_eq!((LOOP_UNSLOTH.len(), LOOP_MASTER.len()), (12, 13));
    }

    /// The table is exact: a stem of each group has its role on a layer of its kind, and a name
    /// the layer's kind does not hold — the other dialect's loop stems, a controller on a
    /// non-core layer or the first layer's mix query, a fused projection, a block past the
    /// stored layers — has none. The loop's message queries carry the core's first layer, the
    /// plan's stage for them. Mutant: every layer given the stems of every kind.
    #[test]
    fn the_table_is_exact() {
        let (pr, master) = (super::Dialect::Unsloth, super::Dialect::Master);
        for name in [
            "token_embd_norm.weight",
            "output_res_query.weight",
            "loop_res_query.weight",
            "loop_kv_res_query.weight",
        ] {
            assert_eq!(
                model_level(name, pr, 2).unwrap(),
                match name {
                    "output_res_query.weight" => (Role::Head, None),
                    "token_embd_norm.weight" => (Role::TokenEmbedding, None),
                    _ => (Role::Attention, Some(2)),
                }
            );
        }
        // The loop's message queries are the unsloth dialect's alone.
        assert!(model_level("loop_res_query.weight", master, 2).is_none());
        assert_eq!(
            role_of("blk.2.attn_rel_bias.weight", pr),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role_of("blk.2.attn_q_inp_norm.weight", pr),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role_of("blk.0.attn_norm.weight", pr),
            Some((Role::Attention, Some(0)))
        );
        assert_eq!(
            role_of("blk.3.attn_res_query.weight", pr),
            Some((Role::Attention, Some(3)))
        );
        assert_eq!(
            role_of("blk.3.ffn_gate_up_exps.weight", pr),
            Some((Role::RoutedExperts, Some(3)))
        );
        assert_eq!(
            role_of("blk.3.exp_probs_b.bias", pr),
            Some((Role::Router, Some(3)))
        );
        assert_eq!(
            role_of("blk.3.loop_attn_down.weight", pr),
            Some((Role::Attention, Some(3)))
        );
        assert_eq!(
            role_of("blk.2.loop_msg_query.weight", master),
            Some((Role::Attention, Some(2)))
        );
        for name in [
            "blk.2.attn_norm.weight",
            "blk.0.attn_q_inp_norm.weight",
            "blk.0.attn_res_query.weight",
            "blk.1.loop_attn_down.weight",
            "blk.3.loop_msg_query.weight",
            "blk.6.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "loop_msg_query.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role_of(name, pr), None, "{name}");
        }
        for name in ["blk.2.loop_attn_gain.weight", "loop_res_query.weight"] {
            assert_eq!(role_of(name, master), None, "{name}");
        }
    }
}
