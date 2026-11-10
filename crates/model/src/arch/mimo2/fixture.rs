//! The MiMo-V2.6-Flash (`mimo2`) gate fixture's spec ([`spec`]): the facts of a `mimo2` file the
//! common generator (`crate::fixture`) cannot know. Every block is a GQA layer of a key head wider
//! than the value head, its KV-head count, window kind and rope base its own; layer 0 runs a dense
//! SwiGLU block and the rest route by a sigmoid router with a selection bias over MXFP4 expert
//! stacks, and the window layers carry attention sinks.
//!
//! The fixture is a full-shape cut of two source layers, renamed `blk.L.` → `blk.f.`, at the
//! source's own widths and counts, so every dispatch class the real header decides is one the
//! fixture runs (`coverage::reach`, `tools/kcover/unreached.tsv`):
//!
//! | fixture layer | source layer | what it is |
//! |---|---|---|
//! | f0 | L0 | a full-attention layer (4 KV heads) with the dense block |
//! | f1 | L1 | a window layer (8 KV heads) with sinks over the routed experts |
//!
//! A full-attention routed layer (L5) adds no class: its attention is L0's and its block L1's.
//!
//! Nothing in the header moves with the map but `block_count` and the two per-layer arrays
//! (`attention.head_count_kv`, `attention.sliding_window_pattern`), taken at the map's layers.
//! [`check_map`] refuses a map that drops a kind or a (stem, type) pair of the source, by name.
//!
//! The fills (the generator's matrix rule is σ 1/√K): the norm gains 1; the sinks and the router's
//! selection bias are spreads, per element ([`SINK_HALF_WIDTH`], [`BIAS_HALF_WIDTH`]), so a body
//! that broadcasts one value or skips one fails.
//!
//! The card budget ([`Budget`]) is planned, not fixed: under it the plan of the written file, at
//! the end-to-end gate's context ([`BUDGET_CTX`]), holds half of the routed experts on the card of
//! every card-eligible layer, and the rest is the host tier's. The budget is a plan's value, not a
//! constant: a plan counts the dense block, the stores, the context, the scratch and the prompt
//! reserve of the file it plans, so a budget below that sum is refused by name at the plan. No
//! draft file: the MTP sidecar is a file of its own, with no fixture yet. No r8 sidecar.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::place::PlanInputs;
use crate::arch::names::blk;
use crate::fixture::hybrid;
use crate::fixture::{
    Budget, CardBudget, CardExperts, Family, FixtureError, FixtureSpec, KeyRule, NoTables, Tables,
    Window,
};
use crate::placement::PlanLevers;
use crate::placement::workstation::{self, HostRead};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 2] = [0, 1];

/// The d window, [2^-14, 2^-4]: normal f16 values every block type's scale fits at every K the
/// fixture holds (2,048 for the down stacks to 16,384 for the dense down).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 16.0;

/// The most tensor data a shard holds, below the fixture's, so the fixture is a split set as the
/// real file is.
pub const SHARD_BYTES: u64 = 4 << 30;

/// The positions the card budget's plan runs at: the end-to-end gate's context
/// (`gate_mimo2_e2e.rs` `CTX`, the 4,096-position set's step plus a margin of whole 64
/// positions). A plan at another context moves the stores, and with them the experts a card holds.
pub const BUDGET_CTX: u64 = 4160;

/// The target file is `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "mimo2-fixture";

/// The real first shard `verify` opens when none is named: the RL file the metadata gate reads
/// (`tests/mimo2_meta.rs`).
pub const DEFAULT_MODEL: &str =
    "/data/moved/models/MiMo-V2.6-Flash-RL/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf";

const ARCH: &str = "mimo2";

/// The half-width of the window layers' sinks (`attn_sinks`), drawn per head: one extra logit per
/// head beside unit-order scores, so a body that drops or broadcasts it moves the softmax.
pub const SINK_HALF_WIDTH: f32 = 1.0;

/// The half-width of the router's selection bias (`exp_probs_b.bias`), drawn per expert: wide
/// enough that it steers the pick, narrow enough that the sigmoid scores still decide it.
pub const BIAS_HALF_WIDTH: f32 = 0.05;

/// The MiMo-V2.6-Flash fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the RL file as
/// the default source, no ff override, no draft and the planned card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Planned(Budget {
            ctx: BUDGET_CTX,
            card_experts,
            card_set_aside: None,
        }),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-4] is inside the normal f16 values"),
        ff: None,
        shard_bytes: SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Mimo2,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map; the two per-layer
/// arrays follow it; `nextn_predict_layers` may be present and must be 0 (the map carries no
/// next-token block).
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 19] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "feed_forward_length",
        "attention.head_count",
        "attention.key_length",
        "attention.value_length",
        "attention.value_scale",
        "attention.layer_norm_rms_epsilon",
        "attention.sliding_window",
        "rope.dimension_count",
        "rope.freq_base",
        "rope.freq_base_swa",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_gating_func",
        "expert_weights_scale",
        "expert_group_count",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count_kv" | "attention.sliding_window_pattern" => KeyRule::PerLayer,
        "nextn_predict_layers" => KeyRule::Zero("the map carries no next-token block"),
        "expert_group_used_count" => KeyRule::Copy,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the norm gains 1.
pub fn const_value(leaf: &str) -> Option<f32> {
    const GAINS: [&str; 3] = ["attn_norm.weight", "ffn_norm.weight", "output_norm.weight"];
    GAINS.contains(&leaf).then_some(1.0)
}

/// The spread of an F32 tensor the family names: the sinks ([`SINK_HALF_WIDTH`]) and the
/// selection bias ([`BIAS_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    match leaf {
        "attn_sinks.weight" => Some(SINK_HALF_WIDTH),
        "exp_probs_b.bias" => Some(BIAS_HALF_WIDTH),
        _ => None,
    }
}

/// A layer's kind as the map keeps it: its window kind, and whether it routes.
type LayerKind = (Kind, bool);

/// The kinds the map keeps one of each: the dense full layer and a routed window layer.
const KEPT: [LayerKind; 2] = [(Kind::Full, false), (Kind::Swa, true)];

/// Every source trunk layer's kind; a layer routes when it holds the router's matrix.
fn kinds(src: &Hparams, source: &Split) -> Vec<LayerKind> {
    (0..src.n_trunk)
        .map(|l| {
            let routes = source.find(&blk(l, "ffn_gate_inp.weight")).is_some();
            (src.kinds[l], routes)
        })
        .collect()
}

/// `layers` keeps the source's facts: one layer of each of [`KEPT`] and every (stem, type) pair any
/// source layer holds ([`hybrid::check_map`]).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    hybrid::check_map(&kinds(src, source), &KEPT, source, layers)
}

/// The per-layer card experts of the plan of `split` on the gate card ([`workstation::plan_gate`],
/// the 3090) at [`BUDGET_CTX`] positions under `budget` bytes, as the end-to-end gate's session
/// plans a file it opens: the file as it is, at the workstation's plan room.
fn card_experts(split: &Split, budget: Option<u64>) -> Result<CardExperts, FixtureError> {
    let mut inputs = PlanInputs::describe(split)?;
    inputs.room = (workstation::PLAN_ROOM, HostRead::Given);
    let machine = workstation::plan_gate(inputs.model.layers);
    let levers = PlanLevers {
        card_budget_bytes: budget.into(),
        ..PlanLevers::default()
    };
    let plan = inputs
        .plan(&machine, BUDGET_CTX, &levers)
        .map_err(|e| FixtureError::Budget(e.to_string()))?;
    Ok(CardExperts {
        per_layer: plan.n_l,
        experts: inputs.model.experts,
    })
}

/// mimo2's rules.
struct Mimo2;

impl Family for Mimo2 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// No dim moves with the map: it is only held to [`check_map`].
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        check_map(&Hparams::read(source)?, source, &spec.layers)?;
        Ok(Box::new(NoTables))
    }

    fn check_kinds(
        &self,
        spec: &FixtureSpec,
        fixture: &Split,
        source: &Split,
    ) -> Result<(), FixtureError> {
        check_kinds(spec, fixture, source).map(|_| ())
    }
}

/// The engine reads `fixture`'s header as the source's hyperparameters at the map's layers: the
/// layer count, each layer's KV-head count and window kind the source layer's, every other fact
/// the source's own, under a map [`check_map`] accepts. Tensor names, dims and types are the plan's,
/// which `verify` holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    let at = |v: &[usize]| spec.layers.iter().map(|&l| v[l]).collect::<Vec<_>>();
    let want = Hparams {
        n_layer: spec.layers.len(),
        n_trunk: spec.layers.len(),
        kv_heads: at(&src.kv_heads),
        kinds: spec.layers.iter().map(|&l| src.kinds[l]).collect(),
        ..src.clone()
    };
    if fx != want {
        return Err(FixtureError::Mismatch {
            what: "the hyperparameters".into(),
            detail: format!("{fx:?}, the cut's {want:?}"),
        });
    }
    Ok(fx)
}

#[cfg(test)]
mod tests {
    use super::{
        BIAS_HALF_WIDTH, KEPT, LAYER_MAP, SINK_HALF_WIDTH, const_value, key_rule, spread_value,
    };
    use crate::fixture::KeyRule;

    /// The map holds one layer of each kept kind by its own table: L0 the dense full layer, L1 a
    /// routed window layer, as the RL file's pattern (`0` at L0, L5, L11, …) has them; the
    /// per-layer arrays follow the map and a next-token count is refused unless 0.
    #[test]
    fn the_keys_and_the_fills() {
        assert_eq!(LAYER_MAP.len(), KEPT.len());
        for key in [
            "attention.head_count_kv",
            "attention.sliding_window_pattern",
        ] {
            assert_eq!(key_rule(key), Some(KeyRule::PerLayer), "{key}");
        }
        assert_eq!(key_rule("block_count"), Some(KeyRule::BlockCount));
        assert!(matches!(
            key_rule("nextn_predict_layers"),
            Some(KeyRule::Zero(_))
        ));
        assert_eq!(key_rule("expert_count"), Some(KeyRule::Copy));
        assert_eq!(key_rule("no_such_key"), None);
        for leaf in ["attn_norm.weight", "ffn_norm.weight", "output_norm.weight"] {
            assert_eq!(const_value(leaf), Some(1.0), "{leaf}");
            assert_eq!(spread_value(leaf), None, "{leaf}");
        }
        assert_eq!(spread_value("attn_sinks.weight"), Some(SINK_HALF_WIDTH));
        assert_eq!(spread_value("exp_probs_b.bias"), Some(BIAS_HALF_WIDTH));
        assert_eq!(const_value("attn_sinks.weight"), None);
        assert_eq!(const_value("ffn_gate_inp.weight"), None);
    }
}
