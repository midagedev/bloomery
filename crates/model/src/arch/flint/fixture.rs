//! The Flint Preview (`flint`) gate fixture's spec ([`spec`]): the facts of a `flint` file the
//! common generator (`crate::fixture`) cannot know. Every block is a GQA layer (32 query heads over
//! 4 key heads of 128) with attention sinks, either windowed (513 positions, rope on) or full
//! (no rope); the first two blocks run a dense SwiGLU block and the other 47 route 6 of 256 experts
//! (a sigmoid router, 2 shared experts), in one unsplit file.
//!
//! The fixture is nine blocks and the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`; each one holds a (tensor stem, type) pair or a layer kind the others do not:
//!
//! | fixture layer | source layer | kind | what only it holds |
//! |---|---|---|---|
//! | f0 | L0 | full, dense | the dense block: Q8_0 `ffn_gate`, `ffn_up`, `ffn_down` |
//! | f1 | L1 | full, dense | the second dense block of the `leading_dense_block_count` |
//! | f2 | L2 | window, routed | `ffn_gate_exps` and `ffn_up_exps` IQ3_S, `ffn_down_exps` Q5_K |
//! | f3 | L4 | window, routed | `ffn_down_exps` Q8_0 beside Q5_K gate and up |
//! | f4 | L8 | window, routed | `ffn_down_exps` Q6_K |
//! | f5 | L19 | window, routed | the Q5_K attention beside IQ4_XS stacks, and Q8_0 shared experts |
//! | f6 | L21 | full, routed | Q4_K `ffn_gate_exps` and `ffn_up_exps` over a full-attention layer |
//! | f7 | L24 | window, routed | Q5_K shared experts beside Q8_0 attention |
//! | f8 | L42 | window, routed | IQ4_XS shared experts |
//!
//! What it keeps: both attention kinds, the dense and the routed block, a routed layer of each
//! attention kind, every (tensor stem, type) pair of the source's layers (the bf16 router, Q8_0
//! and Q5_K attention, Q8_0, Q5_K and IQ4_XS shared experts, and the four routed down and three routed
//! gate and up types), and the real globals (`token_embd` and `output` Q8_0 at their 150,144 rows).
//! What it drops: 40 layers, each of a signature the nine hold. [`check_map`] refuses a map that
//! loses a kind or a pair, by name, at the plan and at `verify`. The map starts with the two dense
//! layers, so `leading_dense_block_count` (2) stays true.
//!
//! Nothing in the header moves with the map but `block_count` and the two per-layer arrays
//! (`attention.sliding_window_pattern`, `attention.rope_pattern`), taken at the map's layers.
//!
//! The fills (the generator's matrix rule is σ 1/√K): the norm gains 1; the sinks are a spread per
//! element ([`SINK_HALF_WIDTH`]), so a body that broadcasts one value or drops them fails. The real
//! sinks run −9 to 12.4 with per-layer means 1.9 to 3.7 [measured on the Q4_K_XL file]; the
//! fixture draws the mechanism, not those values. The router's matrix is the generator's BF16
//! uniform. The card budget ([`Budget`]) is planned, not fixed: under it the plan of the written
//! file, at the lib gate's context ([`BUDGET_CTX`]), holds half of the routed experts on the card of
//! every card-eligible layer, and the rest is the host tier's.

use gguf::Split;

use super::hparams::Hparams;
use super::place::PlanInputs;
use crate::arch::names::blk;
use crate::fixture::hybrid;
use crate::fixture::{
    Budget, CardBudget, CardExperts, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec,
    KeyRule, NoTables, Tables, Window,
};
use crate::placement::PlanLevers;
use crate::placement::workstation::{self, HostRead};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 9] = [0, 1, 2, 4, 8, 19, 21, 24, 42];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixture
/// holds (2,048 for the projections, the shared and routed gate and up stacks and the globals,
/// 512 for the routed down, 4,096 for the attention output, 6,144 for the dense down).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The positions the card budget's plan runs at: the lib gate's context
/// (`crates/gpu-flint/src/tests.rs` `CTX`). A plan at another context moves the stores, and with
/// them the experts a card holds.
pub const BUDGET_CTX: u64 = 1536;

/// The half-width of the sinks (`attn_sinks`), drawn per head: one extra logit per head beside
/// unit-order scores, of the order of the real layers' mean (2 to 4), so a body that drops or
/// broadcasts it moves the softmax.
pub const SINK_HALF_WIDTH: f32 = 3.0;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "flint-fixture";

/// The real file `verify` opens when none is named: the profile's (`tools/ref/models/flint.sh`).
pub const DEFAULT_MODEL: &str = "/models/flint-GGUF/flint-UD-Q4_K_XL.gguf";

const ARCH: &str = "flint";

/// The Flint Preview fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the profile's file
/// as the default source, no ff override, no draft and the planned card budget.
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
        shard_bytes: DEFAULT_SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Flint,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map; the two per-layer
/// arrays follow it, and the dense lead stays because the map starts with the two dense layers.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 18] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "feed_forward_length",
        "leading_dense_block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "attention.sliding_window",
        "rope.freq_base",
        "expert_count",
        "expert_used_count",
        "expert_gating_func",
        "expert_feed_forward_length",
        "expert_shared_count",
        "expert_shared_feed_forward_length",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.sliding_window_pattern" | "attention.rope_pattern" => KeyRule::PerLayer,
        "expert_weights_norm" | "expert_weights_scale" => KeyRule::Copy,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the norm gains 1.
pub fn const_value(leaf: &str) -> Option<f32> {
    const GAINS: [&str; 5] = [
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "output_norm.weight",
    ];
    GAINS.contains(&leaf).then_some(1.0)
}

/// The spread of an F32 tensor the family names: the sinks ([`SINK_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    (leaf == "attn_sinks.weight").then_some(SINK_HALF_WIDTH)
}

/// A layer's kind as the map keeps it: whether it attends a window, and whether it routes.
type LayerKind = (bool, bool);

/// The kinds the map keeps one of each: the dense full layer, a routed window layer and a routed
/// full layer.
const KEPT: [LayerKind; 3] = [(false, false), (true, true), (false, true)];

/// Every source trunk layer's kind; a layer routes when it holds the router's matrix.
fn kinds(src: &Hparams, source: &Split) -> Vec<LayerKind> {
    (0..src.trunk)
        .map(|l| {
            let routes = source.find(&blk(l, "ffn_gate_inp.weight")).is_some();
            (src.window[l].is_some(), routes)
        })
        .collect()
}

/// `layers` keeps the source's facts: one layer of each of [`KEPT`] and every (stem, type) pair any
/// source layer holds ([`hybrid::check_map`]).
pub(super) fn check_map(
    src: &Hparams,
    source: &Split,
    layers: &[usize],
) -> Result<(), FixtureError> {
    hybrid::check_map(&kinds(src, source), &KEPT, source, layers)
}

/// The per-layer card experts of the plan of `split` on the gate card ([`workstation::plan_gate`])
/// at [`BUDGET_CTX`] positions under `budget` bytes, as the lib gate's session plans a file it
/// opens: the file as it is, at the workstation's plan room.
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

/// flint's rules.
struct Flint;

impl Family for Flint {
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
        check_kinds(spec, fixture, source)
    }
}

/// The engine reads `fixture`'s header as the source's hyperparameters at the map's layers: the
/// layer count, each layer's KV-head count, widths, window, rope flag and clamps the source layer's,
/// every other fact the source's own, under a map [`check_map`] accepts. Tensor names, dims and
/// types are the plan's, which `verify` holds the file to. The reader's struct carries no equality,
/// so each field is compared by name.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<(), FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    let n = spec.layers.len();
    let at = |v: &[usize]| spec.layers.iter().map(|&l| v[l]).collect::<Vec<_>>();
    let at_f = |v: &[f32]| spec.layers.iter().map(|&l| v[l]).collect::<Vec<_>>();
    let mut bad: Vec<&str> = Vec::new();
    let mut check = |name: &'static str, same: bool| {
        if !same {
            bad.push(name);
        }
    };
    check("arch", fx.arch == src.arch);
    check("blocks", fx.blocks == n && fx.trunk == n);
    check("next-token blocks", fx.mtp == src.mtp);
    check(
        "widths",
        (fx.hidden, fx.vocab, fx.context) == (src.hidden, src.vocab, src.context),
    );
    check(
        "heads",
        (
            fx.heads,
            fx.head,
            fx.experts,
            fx.used,
            fx.shared_ff,
            fx.dense_lead,
        ) == (
            src.heads,
            src.head,
            src.experts,
            src.used,
            src.shared_ff,
            src.dense_lead,
        ),
    );
    check(
        "scalars",
        (
            fx.eps.to_bits(),
            fx.base.to_bits(),
            fx.scale.to_bits(),
            fx.norm,
            fx.sinks,
        ) == (
            src.eps.to_bits(),
            src.base.to_bits(),
            src.scale.to_bits(),
            src.norm,
            src.sinks,
        ),
    );
    check("kv", fx.kv == at(&src.kv));
    check("ff", fx.ff == at(&src.ff));
    check("expert_ff", fx.expert_ff == at(&src.expert_ff));
    check(
        "window",
        fx.window
            == spec
                .layers
                .iter()
                .map(|&l| src.window[l])
                .collect::<Vec<_>>(),
    );
    check(
        "rope",
        fx.rope == spec.layers.iter().map(|&l| src.rope[l]).collect::<Vec<_>>(),
    );
    check("limit_exp", fx.limit_exp == at_f(&src.limit_exp));
    check("limit_shexp", fx.limit_shexp == at_f(&src.limit_shexp));
    check("limit_dense", fx.limit_dense == at_f(&src.limit_dense));
    if bad.is_empty() {
        Ok(())
    } else {
        Err(FixtureError::Mismatch {
            what: "the hyperparameters".into(),
            detail: format!("{} differ from the source's at the map", bad.join(", ")),
        })
    }
}
