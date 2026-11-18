//! The Nanbeige4.3-33B (`llama`) gate fixture's spec ([`spec`]): the facts of a `llama` file the
//! common generator (`crate::fixture`) cannot know. The model is a dense trunk: every layer is
//! grouped-query attention (56 query heads over 8 key heads of 128) and a SwiGLU block of width
//! 22,016, with no bias, no routed expert and no next-token layer, in one unsplit file.
//!
//! The fixture is four blocks and the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`:
//!
//! | fixture layer | source layer | what the file's layers of its set hold |
//! |---|---|---|
//! | f0 | L0 | the set of L13 to L38: `attn_output` Q6_K, `attn_k` and `attn_v` Q8_0, the rest Q4_K |
//! | f1 | L1 | the set of L1 to L12: `attn_output` Q8_0, `ffn_down` Q5_K, `ffn_up` Q6_K |
//! | f2 | L39 | the set of L39 to L42: `attn_output` Q4_K beside the Q8_0 `attn_k` |
//! | f3 | L51 | the last layer, the one the head reads: `attn_k`, `ffn_down` and `ffn_up` Q6_K, `attn_output` Q5_K |
//!
//! What it keeps: every (tensor stem, type) pair of the source's layers (L43 to L50 add none to
//! these four), the layer-set transitions in file order, and the real globals: `token_embd` (Q4_K)
//! and `output` (Q6_K) keep their 166,912 rows, so the tokenizer's tables and the vocabulary tail
//! are the file's. What it drops: 48 layers, none of them of a type the four do not hold.
//! [`check_map`] refuses a map that loses a pair, by name, at the plan and at `verify`.
//!
//! Nothing in the header moves with the map but `block_count`: the heads, the rope, the epsilon,
//! the vocabulary and the context are the source's. The card budget is fixed: the whole fixture
//! stays resident on a card with room for its cache and scratch.

use gguf::Split;

use super::hparams::{Hparams, RopeMode};
use crate::fixture::hybrid;
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec, KeyRule, NoTables, Tables,
    Window,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [0, 1, 39, 51];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixture
/// holds (7,168 for the projections, the gate and up stacks and the globals, 22,016 for the down).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The card byte budget the fixture records: above the fixture's tensor data (about 3.2 GB), with
/// room for the cache and the scratch of the longest context a gate runs. The model has no routed
/// expert, so nothing in the file plans against it.
pub const CARD_BUDGET: u64 = 6 << 30;

/// The target file of the architecture's fixture is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "llama-fixture";

/// The target file of the `nanbeige43` profile's fixture (`tools/ref/models/nanbeige43.sh`): the
/// same spec under the directory `tools/fixture-dirs.tsv` gives the profile, which the gates read
/// by that name.
pub const PROFILE_STEM: &str = "nanbeige43-fixture";

/// The real file `verify` opens when none is named: the profile's
/// (`tools/ref/models/nanbeige43.sh`).
pub const DEFAULT_MODEL: &str = "/models/Nanbeige4.3-33B/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf";

const ARCH: &str = "llama";

/// The Nanbeige4.3-33B fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the profile's
/// file as the default source, no ff override, no draft and a fixed card budget.
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
        family: &Llama,
    }
}

/// The `nanbeige43` profile's spec: [`spec`] under [`PROFILE_STEM`].
pub fn profile_spec() -> FixtureSpec {
    FixtureSpec {
        stem: PROFILE_STEM,
        ..spec()
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 11] = [
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
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: every norm's gain, 1.
pub fn const_value(leaf: &str) -> Option<f32> {
    const GAINS: [&str; 3] = ["attn_norm.weight", "ffn_norm.weight", "output_norm.weight"];
    GAINS.contains(&leaf).then_some(1.0)
}

/// `layers` keeps the source's facts: every (stem, type) pair any source layer holds
/// ([`hybrid::check_map`], the layers being of one kind).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    hybrid::check_map(&vec![(); src.n_layer], &[()], source, layers)
}

/// llama's rules.
struct Llama;

impl Family for Llama {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    /// No key or dim moves with the map: it is only held to [`check_map`].
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        check_map(
            &Hparams::read(source, RopeMode::Norm)?,
            source,
            &spec.layers,
        )?;
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

/// The engine reads `fixture`'s header as the source's hyperparameters at the map's layer count,
/// under a map [`check_map`] accepts. Tensor names, dims and types are the plan's, which `verify`
/// holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture, RopeMode::Norm)?;
    let src = Hparams::read(source, RopeMode::Norm)?;
    check_map(&src, source, &spec.layers)?;
    let want = Hparams {
        n_layer: spec.layers.len(),
        ..src
    };
    if fx != want {
        return Err(FixtureError::Mismatch {
            what: "the hyperparameters".into(),
            detail: format!("{fx:?}, the source's at the map {want:?}"),
        });
    }
    Ok(fx)
}
