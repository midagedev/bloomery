//! The Solar-Open2-250B (`solar-open2`) gate fixture's spec ([`spec`]): the facts of a
//! `solar-open2` file the common generator (`crate::fixture`) cannot know. The model is a hybrid
//! of Kimi delta attention (KDA) layers and NoPE GQA layers (a GQA layer at every fourth index),
//! every layer routing 320 experts of 1,280 beside one shared expert, and the file is a split
//! set.
//!
//! The fixture is four blocks and the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`; no expert is dropped:
//!
//! | fixture layer | source layer | mixer | `attn_v`, `ffn_down_exps`, `ffn_down_shexp` | why |
//! |---|---|---|---|---|
//! | f0 | L0 | GQA | Q6_K | the file's first layer: the first GQA layer with the Q6_K projections |
//! | f1 | L1 | KDA | Q6_K | the first KDA layer, after a GQA layer, with the Q6_K projections |
//! | f2 | L6 | KDA | Q4_K | the first KDA layer with the Q4_K projections, after a KDA layer |
//! | f3 | L12 | GQA | Q4_K | the first GQA layer with the Q4_K projections, after a KDA layer |
//!
//! What it keeps: both mixers, each with both type sets the source's layers hold (the quantizer
//! gives a layer the Q6_K or the Q4_K set, whichever the layer's index puts it in), the three
//! transitions between them (GQA→KDA, KDA→KDA, KDA→GQA), every (tensor stem, type) pair of the
//! source's layers, and the 320-expert top-8 router of every layer. What it drops: 33 KDA layers
//! and 11 GQA layers. [`check_map`] refuses a map that loses a kind or a pair, by name, at the
//! plan and at `verify`.
//!
//! The tensor data is 13,651,122,688 bytes in 4 shards [derived from the header's tensor sizes]:
//! 1.11 GB of it the globals (`token_embd` Q4_K, `output` Q6_K), 12.54 GB the four layers. A
//! shard holds at most [`SHARD_BYTES`], so a layer spans a shard boundary, as it does in the
//! source's four shards.
//!
//! Nothing in the header moves with the map but `block_count` and the per-layer
//! `attention.head_count_kv` array (0 on a KDA layer): the experts, the heads, the conv taps and
//! the vocabulary are copied. The reader refuses a dense lead and a next-token block, so the
//! copied `leading_dense_block_count` and `nextn_predict_layers` of a source it reads are 0.
//!
//! The card budget ([`Budget`]) is planned, not fixed: under it the plan of the written file, at
//! the end-to-end gate's context, holds half of the experts, 160 of 320, on every card-eligible
//! layer (f2 and f3), and the other half is the host tier's. The Q6_K-down layers (f0 and f1) are
//! not card-eligible, since no card kernel runs that stack: they keep all their experts on the
//! host, as the source's Q6_K layers do.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::place::PlanInputs;
use crate::arch::hybrid::{ATTN_NORM, SSM_A, SSM_DT_BIAS, SSM_NORM};
use crate::fixture::hybrid;
use crate::fixture::{
    Budget, CardBudget, CardExperts, Family, FixtureError, FixtureSpec, KeyRule, NoTables, Tables,
    Window,
};
use crate::placement::PlanLevers;
use crate::placement::workstation::{self, HostRead};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [0, 1, 6, 12];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixture
/// holds (K from 4 for the conv taps to 8,192 for `attn_output`; the Q5_0 low-rank halves at
/// K = 128 and the Q6_K down stack at K = 1,280 are the narrowest).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The positions the card budget's plan runs at: the end-to-end gate's context, the longest
/// prompt and the room beyond it (`gate_solar2_e2e.rs`). A plan at another context moves the
/// stores, and with them the experts a card holds.
pub const BUDGET_CTX: u64 = 1552;

/// The most tensor data a shard holds. The source is a four-shard split set: a cap of 4 GiB cuts
/// the fixture's 13.7 GB into four shards with a layer across a boundary, so the engine reads a
/// split set here too.
pub const SHARD_BYTES: u64 = 4 << 30;

/// The target shards are `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "solar2-fixture";

/// The real first shard `verify` opens when none is named: the profile's
/// (`tools/ref/models/solar-open2.sh`).
pub const DEFAULT_MODEL: &str =
    "/models/Solar-Open2-250B-Q4_K_M/Solar-Open2-250B-Q4_K_M-00001-of-00004.gguf";

const ARCH: &str = "solar-open2";

/// The `ssm_a` constant: the median over a layer's 64 heads, taken over the 36 KDA layers, is
/// −1.49 [measured; the layers' medians run from −1.54 to −1.44], so the decay
/// `ssm_a · softplus(x + dt)` is about the real file's for a unit-scale `x`.
const SSM_A_VALUE: f32 = -1.49;

/// The `ssm_dt.bias` constant: the median over the 36 KDA layers' 8,192 channels is −0.84
/// [measured; the layers' medians run from −0.85 to −0.83].
const SSM_DT_BIAS_VALUE: f32 = -0.84;

/// The router's selection bias, by its name past `blk.N.`.
const EXP_PROBS_B: &str = "exp_probs_b.bias";

/// The half-width of the router's selection bias (`exp_probs_b.bias`), drawn per expert. A
/// constant bias moves no pick, so a body that skipped the add would pass. The real file's
/// biases have a standard deviation of 0.056 at the median over its 48 layers (0.020 to 0.083),
/// and a uniform draw in ±0.1 has 0.058, so the add moves picks as it does in the real file.
pub const BIAS_HALF_WIDTH: f32 = 0.1;

/// The Solar-Open2 fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the profile's file
/// as the default source, the experts' own width, no draft (the file has no next-token layer) and
/// the planned card budget.
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
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        ff: None,
        shard_bytes: SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Solar,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map, and the keys the
/// file carries that no layer reads (`feed_forward_length`, `rope.freq_base`).
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 24] = [
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "vocab_size",
        "attention.head_count",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "kda.head_dim",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_count",
        "expert_gating_func",
        "expert_weights_scale",
        "expert_weights_norm",
        "expert_group_count",
        "expert_group_used_count",
        "leading_dense_block_count",
        "nextn_predict_layers",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count_kv" => KeyRule::PerLayer,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the norms' gains 1,
/// the KDA decay's `ssm_a` and `ssm_dt.bias` ([`SSM_A_VALUE`], [`SSM_DT_BIAS_VALUE`]). The router's
/// selection bias is a spread ([`spread_value`]).
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 6] = [
        (ATTN_NORM, 1.0),
        ("ffn_norm.weight", 1.0),
        ("output_norm.weight", 1.0),
        (SSM_NORM, 1.0),
        (SSM_A, SSM_A_VALUE),
        (SSM_DT_BIAS, SSM_DT_BIAS_VALUE),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The spread of an F32 tensor the family names: the router's selection bias
/// ([`BIAS_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    (leaf == EXP_PROBS_B).then_some(BIAS_HALF_WIDTH)
}

/// `layers` keeps the source's facts: one layer of each mixer and every (stem, type) pair any
/// source layer holds ([`hybrid::check_map`]).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    let kinds: Vec<Kind> = (0..src.n_layer).map(|l| src.kind(l)).collect();
    hybrid::check_map(&kinds, &[Kind::Kda, Kind::Gqa], source, layers)
}

/// The written file's per-layer card expert counts, planned through the placement planner on the
/// gate card ([`workstation::plan_gate`], the 3090) at [`BUDGET_CTX`] positions under `budget`
/// bytes, as the end-to-end gate's seat plans a file it opens: the file as it is, at the
/// workstation's plan room.
fn card_experts(split: &Split, budget: Option<u64>) -> Result<CardExperts, FixtureError> {
    let mut inputs = PlanInputs::describe(split)?;
    inputs.room = (workstation::PLAN_ROOM, HostRead::Given);
    let machine = workstation::plan_gate(inputs.model.layers);
    let levers = PlanLevers {
        card_budget_bytes: budget.into(),
    };
    let plan = inputs
        .plan(&machine, BUDGET_CTX, &levers)
        .map_err(|e| FixtureError::Budget(e.to_string()))?;
    Ok(CardExperts {
        per_layer: plan.n_l,
        experts: inputs.model.experts,
    })
}

/// solar-open2's rules.
struct Solar;

impl Family for Solar {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// No key moves with the map but the two [`KeyRule`]s above, so the family has no tables;
    /// the map is held to [`check_map`].
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

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds and KV head counts are the
/// source layers', at the source's globals, KDA numbers and routed block, under a map
/// [`check_map`] accepts. Tensor names, dims and types are the plan's, which `verify` holds the
/// file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    if fx.n_layer != spec.layers.len() {
        return Err(bad(
            "layer count".into(),
            format!("{}, not {}", fx.n_layer, spec.layers.len()),
        ));
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        if fx.kind(f) != src.kind(l) {
            return Err(bad(
                what,
                format!("kind {:?}, the source's {:?}", fx.kind(f), src.kind(l)),
            ));
        }
        if fx.kv_heads[f] != src.kv_heads[l] {
            return Err(bad(
                what,
                format!(
                    "kv heads {}, the source's {}",
                    fx.kv_heads[f], src.kv_heads[l]
                ),
            ));
        }
    }
    for (key, a, b) in [
        ("n_embd", fx.n_embd, src.n_embd),
        ("n_head", fx.n_head, src.n_head),
        ("head_dim", fx.head_dim, src.head_dim),
        ("n_vocab", fx.n_vocab, src.n_vocab),
        ("n_ctx_train", fx.n_ctx_train, src.n_ctx_train),
        ("ssm.conv_kernel", fx.kda.conv, src.kda.conv),
        ("kda.head_dim", fx.kda.head_dim, src.kda.head_dim),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, want {b}")));
        }
    }
    if fx.rms_eps.to_bits() != src.rms_eps.to_bits() {
        return Err(bad(
            "attention.layer_norm_rms_epsilon".into(),
            format!("{}, the source's {}", fx.rms_eps, src.rms_eps),
        ));
    }
    if fx.experts != src.experts {
        return Err(bad(
            "the routed block".into(),
            format!("{:?}, the source's {:?}", fx.experts, src.experts),
        ));
    }
    if fx.defaults != src.defaults {
        return Err(bad(
            "the keys read by default".into(),
            format!("{:?}, the source's {:?}", fx.defaults, src.defaults),
        ));
    }
    Ok(fx)
}
