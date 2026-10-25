//! The Qwen3-30B-A3B-Instruct-2507 (`qwen3moe`) gate fixture's spec ([`spec`]): the facts of a
//! `qwen3moe` file the common generator (`crate::fixture`) cannot know. The model is a uniform
//! trunk: every layer is grouped-query attention with per-head Q and K norms over a router that
//! picks 8 of 128 experts, no shared expert, no dense layer, and the file is one unsplit file.
//!
//! The fixture is four blocks and the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`; no expert is dropped:
//!
//! | fixture layer | source layer | `attn_v`, `ffn_down_exps` | why |
//! |---|---|---|---|
//! | f0 | L6 | Q4_K | the file's first layer with the Q4_K pair (L0–L5 hold the Q6_K one) |
//! | f1 | L8 | Q6_K | the first Q6_K layer after a Q4_K one: the Q4_K→Q6_K transition |
//! | f2 | L9 | Q4_K | the Q6_K→Q4_K transition |
//! | f3 | L47 | Q6_K | the last layer, the one the head reads, with the Q6_K pair |
//!
//! What it keeps: both type sets the quantizer gives a layer (the `use_more_bits` rule puts a
//! layer in the Q6_K set or the Q4_K set by its index), both transitions between them, every
//! (tensor stem, type) pair of the source's layers, the 128-expert top-8 router of every layer,
//! and the real globals: `token_embd` (Q4_K) and `output` (Q6_K) keep their 151,936 rows, so the
//! vocabulary tail and the tokenizer tables are the file's. What it drops: 44 layers, none of
//! them of a kind or a type the four do not hold. [`check_map`] refuses a map that loses a pair,
//! by name, at the plan and at `verify`.
//!
//! The tensor data is 1,940,326,400 bytes [derived from the header's dims and the block sizes
//! of Q4_K (144 B per 256), Q6_K (210 B) and F32]: 430,290,944 of it the globals, 351,421,440
//! each of the two Q4_K layers and 403,596,288 each of the two Q6_K ones. It is one shard
//! ([`DEFAULT_SHARD_BYTES`] is above it), as the source is one file.
//!
//! Nothing in the header moves with the map but `block_count`: the heads, the experts, the rope,
//! the epsilon and the context are the source's. The router's selection is a softmax over
//! the experts' logits with no bias tensor, so no spread is drawn; `ffn_gate_inp` is the matrix
//! every F32 matrix is, σ 1/√K.
//!
//! The card budget is fixed, not planned: the recorded value keeps the whole fixture resident
//! on a card with room for its cache and scratch, and the engine's attach sets the value its
//! placement reads. No draft file (the real file holds no next-token tensors) and no r8 sidecar.

use gguf::Split;

use super::hparams::Hparams;
use crate::fixture::hybrid;
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec, KeyRule, NoTables, Tables,
    Window,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [6, 8, 9, 47];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixture
/// holds (768 for the down stacks, 2,048 for the projections, the gate and up stacks and the
/// globals, 4,096 for `attn_output`).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The card byte budget the fixture records: above the fixture's tensor data (1.94 GB), with room
/// for the cache and the scratch of the longest context a gate runs.
pub const CARD_BUDGET: u64 = 4 << 30;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "qwen3moe-fixture";

/// The real file `verify` opens when none is named: the profile's
/// (`tools/ref/models/qwen3moe.sh`).
pub const DEFAULT_MODEL: &str = "/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";

const ARCH: &str = "qwen3moe";

/// The Qwen3-30B-A3B fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the profile's file
/// as the default source, no ff override, no draft and a fixed card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Fixed(CARD_BUDGET),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        ff: None,
        shard_bytes: DEFAULT_SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Qwen3,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map, and the key the
/// file carries that no layer reads (`feed_forward_length`).
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 20] = [
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "vocab_size",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "rope.dimension_count",
        "rope.scaling.type",
        "rope.scaling.factor",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_count",
        "expert_gating_func",
        "expert_weights_norm",
        "expert_weights_scale",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: every norm's gain, 1
/// (the two per-head norms of a layer, `attn_q_norm` and `attn_k_norm`, included).
pub fn const_value(leaf: &str) -> Option<f32> {
    const GAINS: [&str; 5] = [
        "attn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_norm.weight",
        "output_norm.weight",
    ];
    GAINS.contains(&leaf).then_some(1.0)
}

/// `layers` keeps the source's facts: every (stem, type) pair any source layer holds
/// ([`hybrid::check_map`], the layers being of one kind).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    hybrid::check_map(&vec![(); src.n_layer], &[()], source, layers)
}

/// The facts of `hp` a fixture keeps, by the name a refusal gives them.
fn facts(hp: &Hparams) -> [(&'static str, String); 8] {
    [
        ("embedding_length", hp.n_embd.to_string()),
        ("attention.head_count", hp.n_head.to_string()),
        ("attention.head_count_kv", hp.n_head_kv.to_string()),
        ("attention head width", hp.head_dim.to_string()),
        ("the rope", format!("{:?}", hp.rope)),
        (
            "attention.layer_norm_rms_epsilon",
            format!("{:?}", hp.rms_eps),
        ),
        (
            "the vocabulary and context",
            format!("{} and {}", hp.n_vocab, hp.n_ctx_train),
        ),
        ("the routed block", format!("{:?}", hp.experts)),
    ]
}

/// qwen3moe's rules.
struct Qwen3;

impl Family for Qwen3 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    /// No key or dim moves with the map: it is only held to [`check_map`].
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

/// The engine reads `fixture`'s header as `spec`'s layer count at the source's heads, rope,
/// epsilon, vocabulary, context and routed block, under a map [`check_map`] accepts. Tensor names,
/// dims and types are the plan's, which `verify` holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    if fx.n_layer != spec.layers.len() {
        return Err(FixtureError::Mismatch {
            what: "layer count".into(),
            detail: format!("{}, not {}", fx.n_layer, spec.layers.len()),
        });
    }
    for ((key, a), (_, b)) in facts(&fx).into_iter().zip(facts(&src)) {
        if a != b {
            return Err(FixtureError::Mismatch {
                what: key.into(),
                detail: format!("{a}, the source's {b}"),
            });
        }
    }
    Ok(fx)
}
