//! The LFM3-1.7B (`lfm3`) gate fixture's spec ([`spec`]): the facts of an `lfm3` file the common
//! generator (`crate::fixture`) cannot know. The trunk mixes gated short-conv layers and
//! grouped-query attention layers (sliding window or full) under a dense SwiGLU block of width
//! 7,680 in every layer, sandwich norms around both sublayers and a head tied to the token
//! embedding, in one unsplit file.
//!
//! The fixture is four blocks and the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`:
//!
//! | fixture layer | source layer | kind | why |
//! |---|---|---|---|
//! | f0 | L0 | short conv | the conv layers of the Q6_K `ffn_down` set (L0, L1, L5, L11, L17, L21, L23) |
//! | f1 | L2 | full attention | the attention layers of the Q6_K `attn_v` and `ffn_down` set (L2, L8, L14, L20, L22) |
//! | f2 | L3 | short conv | the conv layers of the Q4_K `ffn_down` set |
//! | f3 | L4 | sliding attention | the attention layers of the Q4_K `attn_v` set, and the first sliding one |
//!
//! What it keeps: one layer of each kind (short conv, sliding, full), every (tensor stem, type)
//! pair of the source's layers, and the real globals (`token_embd` Q6_K at its 128,000 rows, the
//! rope factor table). What it drops: 20 layers, none of a kind or a type the four do not hold.
//! [`check_map`] refuses a map that loses a kind or a pair, by name, at the plan and at `verify`.
//!
//! The two per-layer arrays (`attention.head_count_kv`, `attention.sliding_window_pattern`) follow
//! the map. The fills: every norm's gain 1, and the rope factor table 1 everywhere: the real
//! table holds 1 for its first 32 pairs and 1e30 for the last 32 (a pair that does not turn),
//! which the generator's one-value rule cannot hold, so the clause that reads those values stays
//! on the real file. The card budget is fixed: the model is 1.7 B parameters and no program
//! plans against it.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use crate::fixture::hybrid;
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec, KeyRule, NoTables, Tables,
    Window,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [0, 2, 3, 4];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixture
/// holds (2,048 for the projections and the embedding, 7,680 for the down).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The card byte budget the fixture records: far above the fixture's tensor data (0.35 GB).
pub const CARD_BUDGET: u64 = 2 << 30;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "lfm3-fixture";

/// The real file `verify` opens when none is named: the profile's (`tools/ref/models/lfm3.sh`).
pub const DEFAULT_MODEL: &str = "/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf";

const ARCH: &str = "lfm3";

/// The kinds the map keeps one of each.
const KEPT: [Kind; 3] = [Kind::Conv, Kind::Sliding, Kind::Full];

/// The LFM3-1.7B fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the profile's file as
/// the default source, no ff override, no draft and a fixed card budget.
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
        family: &Lfm3,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map; the per-layer
/// keys follow the map.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 13] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "attention.key_length",
        "attention.value_length",
        "attention.sliding_window",
        "shortconv.l_cache",
        "rope.dimension_count",
        "rope.freq_base",
        "rope.freq_base_swa",
        "rope.scaling.finetuned",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count_kv" | "feed_forward_length" | "attention.sliding_window_pattern" => {
            KeyRule::PerLayer
        }
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.` (or the whole name of a
/// global): every norm's gain, and the rope factor table, 1.
pub fn const_value(leaf: &str) -> Option<f32> {
    const ONES: [&str; 8] = [
        "attn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "post_attention_norm.weight",
        "ffn_norm.weight",
        "post_ffw_norm.weight",
        "output_norm.weight",
        "rope_freqs.weight",
    ];
    ONES.contains(&leaf).then_some(1.0)
}

/// `layers` keeps the source's facts: one layer of each of [`KEPT`] and every (stem, type) pair
/// any source layer holds ([`hybrid::check_map`]).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    hybrid::check_map(&src.kinds, &KEPT, source, layers)
}

/// lfm3's rules.
struct Lfm3;

impl Family for Lfm3 {
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

/// The engine reads `fixture`'s header as the source's hyperparameters at the map's layers: the
/// layer count, each layer's KV-head count, feed-forward width and kind the source layer's, every
/// other fact the source's own, under a map [`check_map`] accepts. Tensor names, dims and types
/// are the plan's, which `verify` holds the file to.
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
        kv: at(&src.kv),
        ff: at(&src.ff),
        kinds: spec.layers.iter().map(|&l| src.kinds[l]).collect(),
        ..src.clone()
    };
    if fx != want {
        return Err(FixtureError::Mismatch {
            what: "the hyperparameters".into(),
            detail: format!("{fx:?}, the source's at the map {want:?}"),
        });
    }
    Ok(fx)
}
