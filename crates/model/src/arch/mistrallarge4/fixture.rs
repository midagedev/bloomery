//! The Mistral Large 4 (`mistral-large4`) gate fixture's spec ([`spec`]): the facts of a
//! `mistral-large4` file the common generator (`crate::fixture`) cannot know. The model is one
//! repeated layer — latent attention with no query latent, SSMax on the query, the lightning
//! indexer picking the positions, a softmax-routed mixture of experts beside a shared one, the
//! first layer dense — and the file is a split set whose first shard holds the metadata alone.
//!
//! The fixture is a requantized mini-Mistral of five real layers, renamed `blk.L.` → `blk.f.`:
//!
//! | fixture layer | source layer | what it is |
//! |---|---|---|
//! | f0 | L0 | the dense lead: roped, a full indexer (the reader refuses a shared first), its own cache |
//! | f1 | L1 | the first routed layer: roped, a full indexer, its own cache |
//! | f2 | L17 | roped, a shared indexer: it reuses f1's list and keys and carries none of the nine indexer tensors |
//! | f3 | L35 | NoPE, a full indexer, its own cache: the owner the reader reads (a cache row of the latent alone) |
//! | f4 | L39 | NoPE, a full indexer, reading f3's cache: no `attn_kv_a_mqa`, no `attn_kv_a_norm` |
//!
//! The per-layer arrays are the source's own at the map's layers — `attention.rope_pattern`
//! `[T,T,T,F,F]`, `attention.indexer.types` `[T,T,F,T,T]` — and `attention.kv_source_map`, whose
//! values are layer ids, is renumbered to `[0,1,2,3,3]` ([`KeyRule::PerLayerIds`]). The map is
//! read from the source's arrays, layer by layer ([`KINDS`]), never reconstructed from a rule.
//!
//! What it keeps: the latent 512, the rope width 64 and with it the head's `key_length_mla` 128
//! and both cache-row widths (576 on a roped layer, 512 on a NoPE one), the index heads 32 × 128
//! (the score kernel's own constant), the router's top-4 over its rivals, the shared expert, the
//! YaRN keys and the LeRoPE factors' 32 values, the head's softcap, and the real vocabulary: the
//! globals keep their second axis, and the tokenizer tables are the source's copied ones, so the
//! fixture's tokenizer is the file's. Every (tensor stem, type) pair of the five real layers is
//! kept — Q6_K `attn_q`, `attn_output`, `indexer.attn_q_b` and `output`/`token_embd`, Q8_0 the
//! latent, its up-projections, the indexer's key, the dense lead and the shared expert, IQ2_XXS
//! and IQ1_S the routed stacks as the layer's own — and F32 the norms, the SSMax pairs, the
//! router and the indexer's head weights.
//!
//! What it minifies: the hidden (1,024), the head count (16), the routed experts (16 of width
//! 256, whose shared expert is one of the same width), the dense lead's width (4,096, the real
//! file's 4 × hidden) and the indexer's `top_k` (16), which the reader serves a context up to
//! ([`super::spec::serve_ctx`]): a prompt past it selects, where the real file's 2,048 selects
//! past a context of that length. Every width is a whole block of every type its tensors hold.
//! The shapes are the reader's own: the dims of each source tensor are what the reader asks of
//! a file of the mini's keys at the source layer's kind (`Hparams::want`), so a stem the reader
//! reads and the fixture has no table for cannot exist, and the file the plan writes reads back
//! through the reader's own dims check.
//!
//! The fills (the generator's matrix rule is σ 1/√K, so a projection of a unit-RMS input has unit
//! RMS): the norm gains are 1 but the indexer's LayerNorm gains ([`INDEXER_Q_GAIN`],
//! [`INDEXER_K_GAIN`]) and the rope's frequency factors ([`LEROPE_FACTOR`]), each a value that a
//! body skipping it, or multiplying where the file divides, reads the oracle wrong at; the
//! indexer's LayerNorm biases and the SSMax pairs are spreads ([`LN_BIAS_HALF_WIDTH`],
//! [`SSMAX_HALF_WIDTH`]), per element, so a body that broadcasts one value or skips one fails.
//!
//! The fixture is one shard of 119 tensors: 415,962,880 bytes of tensor data, 344,068,224 of it
//! the four globals and the rest the five layers, in a file of 430,204,832 bytes [measured:
//! `fixture plan`; the 14,241,952 header bytes are the source's tokenizer tables and the keys].
//!
//! The card budget is fixed, not planned: the architecture has no program yet, so no plan of the
//! written file chooses a cap; the recorded 2 GiB keeps the whole fixture resident. No draft file
//! (the DSpark stages ship in a file of their own) and no r8 sidecar.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{
    ATTN_KV_A_NORM, ATTN_NORM, ATTN_SSMAX_B, ATTN_SSMAX_W, FFN_NORM, INDEXER_K_NORM_B,
    INDEXER_K_NORM_W, INDEXER_Q_NORM_B, INDEXER_Q_NORM_W, INDEXER_SSMAX_B, INDEXER_SSMAX_W,
};
use crate::arch::classify::ROPE_FREQS;
use crate::arch::names::{blk, output, output_norm, token_embd};
use crate::fixture::{CardBudget, Family, FixtureError, FixtureSpec, KeyRule, Tables, Window};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 5] = [0, 1, 17, 35, 39];

/// The hidden the mini writes: `embedding_length`, the query field and the residual stream.
pub const HIDDEN: u64 = 1024;
/// The query heads the mini writes; the head's width stays the real file's 128.
pub const HEADS: u64 = 16;
/// The routed experts the mini writes: `expert_count`.
pub const EXPERTS: u64 = 16;
/// One routed expert's width the mini writes: `expert_feed_forward_length`, which the shared
/// expert's (`expert_shared_count` of them) shares, whole blocks of every type of its tensors.
pub const FF: u64 = 256;
/// The dense lead's width the mini writes: `feed_forward_length`, the real file's 4 × hidden.
pub const DENSE_FF: u64 = 4096;
/// The indexer's list length the mini writes: `attention.indexer.top_k`.
pub const TOP_K: u64 = 16;

/// The d window, [2^-14, 2^-4]: normal f16 values every type's scale fits at every K the mini
/// holds. The floor is the smallest normal f16 and binds the Q6_K stacks (σ 1/√K needs d of
/// 2^-14.0 at K = 1,024 with the scale band cut to 47 of 127, and at K = 2,048 to 33) [measured:
/// `fixture plan`'s rule lines]; the top binds nothing, the largest d drawn being the IQ1_S down
/// stack's, about 2^-6.9 at K = 256 [derived: σ over `√(E[(2s+1)²]·(E[g²]+δ²))`, ≈ 7.2].
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 16.0;

/// The most tensor data a shard holds: the shared default, above the fixture's tensor data, so
/// the fixture is one shard.
pub const SHARD_BYTES: u64 = crate::fixture::DEFAULT_SHARD_BYTES;

/// The card byte budget the fixture records. Above the fixture's tensor data, it keeps the whole
/// model resident; the architecture has no program to plan a cap through.
pub const CARD_BUDGET: u64 = 2 << 30;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "mistrallarge4-fixture";

/// The real first shard `verify` opens when none is named: the profile's
/// (`tools/ref/models/mistral-large4.sh`).
pub const DEFAULT_MODEL: &str =
    "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf";

const ARCH: &str = "mistral-large4";

/// The gain the mini writes into the indexer's query LayerNorm (`indexer.q_norm.weight`): 2,
/// not 1, so a body that skips the gain still reads the oracle wrong at the indexer's query.
pub const INDEXER_Q_GAIN: f32 = 2.0;

/// The gain the mini writes into the indexer's key LayerNorm (`indexer.k_norm.weight`): 1/2,
/// the same reason, and a pair whose scores a body that swaps the two still reads wrong.
pub const INDEXER_K_GAIN: f32 = 0.5;

/// The factor the mini writes into every one of the rope's 32 frequency factors
/// (`rope_freqs.weight`): the rope divides each frequency by it, so 2 halves them all, and a
/// body that skips the factors, or multiplies where the file divides, reads the oracle wrong.
/// The real file's factors differ per frequency; that spread is the real tier's.
pub const LEROPE_FACTOR: f32 = 2.0;

/// The half-width of the indexer's LayerNorm biases (`indexer.q_norm.bias`,
/// `indexer.k_norm.bias`), drawn per element. A LayerNorm's output has unit variance, so a
/// uniform ±0.5 is an RMS of 0.29 beside it [derived: 0.5/√3]: skipping the bias is a 29 % error
/// at the projection's input, not a rounding.
pub const LN_BIAS_HALF_WIDTH: f32 = 0.5;

/// The half-width of the SSMax scale and bias (`attn_ssmax.*`, `indexer.ssmax.*`), drawn per
/// head. The query is multiplied by `s·ln(pos + 1) + b`, whose RMS at position 17 is
/// `h/√3·√(ln²18 + 1)` ≈ 0.9 for h = 0.5 [derived], a scale of order one that differs per head
/// and changes sign, so a body that broadcasts one value, or skips the pair, fails.
pub const SSMAX_HALF_WIDTH: f32 = 0.5;

/// What a source layer must read as for the map to keep a fixture layer's kind: the rows of the
/// table above.
struct Kind {
    /// The layer ropes.
    rope: bool,
    /// The layer runs its own indexer.
    full_indexer: bool,
    /// The fixture layer whose cache the layer reads (itself when it owns one).
    cache: usize,
    /// The layer is the dense lead.
    dense: bool,
}

/// The five kinds of [`LAYER_MAP`], layer by layer.
const KINDS: [Kind; 5] = [
    Kind {
        rope: true,
        full_indexer: true,
        cache: 0,
        dense: true,
    },
    Kind {
        rope: true,
        full_indexer: true,
        cache: 1,
        dense: false,
    },
    Kind {
        rope: true,
        full_indexer: false,
        cache: 2,
        dense: false,
    },
    Kind {
        rope: false,
        full_indexer: true,
        cache: 3,
        dense: false,
    },
    Kind {
        rope: false,
        full_indexer: true,
        cache: 3,
        dense: false,
    },
];

/// The Mistral Large 4 fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], no default
/// source, no ff override (the mini's expert width is a header key, not an
/// ff axis), no draft and a fixed card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Fixed(CARD_BUDGET),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-4] is inside the normal f16 values"),
        ff: None,
        shard_bytes: SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Mini,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. What the
/// mini resizes is a [`KeyRule::Set`]; the three per-layer arrays follow the map, the cache map
/// renumbered; the rest — the epsilons, the head, latent and rope widths, the YaRN keys, the
/// index heads, the router's constants, the dense-block count and the softcap — is copied.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: &[&str] = &[
        "context_length",
        "vocab_size",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.key_length_mla",
        "attention.value_length_mla",
        "attention.kv_lora_rank",
        "attention.q_lora_rank",
        "attention.layer_norm_rms_epsilon",
        "attention.layer_norm_epsilon",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "rope.dimension_count",
        "rope.freq_base",
        "rope.scaling.type",
        "rope.scaling.factor",
        "rope.scaling.original_context_length",
        "rope.scaling.yarn_beta_fast",
        "rope.scaling.yarn_beta_slow",
        "rope.scaling.yarn_log_multiplier",
        "leading_dense_block_count",
        "final_logit_softcapping",
        "expert_used_count",
        "expert_shared_count",
        "expert_gating_func",
        "expert_group_count",
        "expert_group_used_count",
        "expert_weights_norm",
        "expert_weights_scale",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.rope_pattern" | "attention.indexer.types" => KeyRule::PerLayer,
        "attention.kv_source_map" => KeyRule::PerLayerIds,
        "embedding_length" => KeyRule::Set(HIDDEN),
        "feed_forward_length" => KeyRule::Set(DENSE_FF),
        "attention.head_count" => KeyRule::Set(HEADS),
        "expert_count" => KeyRule::Set(EXPERTS),
        "expert_feed_forward_length" => KeyRule::Set(FF),
        "attention.indexer.top_k" => KeyRule::Set(TOP_K),
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the norm gains 1, the
/// indexer's LayerNorm gains ([`INDEXER_Q_GAIN`], [`INDEXER_K_GAIN`]) and the rope's factors
/// ([`LEROPE_FACTOR`]). The LayerNorm biases and the SSMax pairs are spreads ([`spread_value`]).
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 7] = [
        (ATTN_NORM, 1.0),
        (ATTN_KV_A_NORM, 1.0),
        (FFN_NORM, 1.0),
        (INDEXER_Q_NORM_W, INDEXER_Q_GAIN),
        (INDEXER_K_NORM_W, INDEXER_K_GAIN),
        // The globals' leaves are spelled as the common name helper does: their names are its
        // functions, not constants.
        ("output_norm.weight", 1.0),
        (ROPE_FREQS, LEROPE_FACTOR),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The spread of an F32 tensor the family names: the indexer's LayerNorm biases
/// ([`LN_BIAS_HALF_WIDTH`]) and the SSMax scale and bias of the attention and of the indexer
/// ([`SSMAX_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    match leaf {
        INDEXER_Q_NORM_B | INDEXER_K_NORM_B => Some(LN_BIAS_HALF_WIDTH),
        ATTN_SSMAX_W | ATTN_SSMAX_B | INDEXER_SSMAX_W | INDEXER_SSMAX_B => Some(SSMAX_HALF_WIDTH),
        _ => None,
    }
}

/// The map's layers must read as [`KINDS`] in `src`, layer by layer from the source's own
/// arrays, and the dense lead must be the first of them alone. A source that differs is refused
/// by name, so a rule of the model that moved cannot hide behind a map that still has the right
/// layer numbers.
fn check_map(src: &Hparams, layers: &[usize]) -> Result<(), FixtureError> {
    if layers.len() != KINDS.len() {
        return Err(FixtureError::Mismatch {
            what: "the layer map".into(),
            detail: format!(
                "holds {} layers, the fixture is built for the {} of the kinds above",
                layers.len(),
                KINDS.len()
            ),
        });
    }
    for (f, (&l, kind)) in layers.iter().zip(&KINDS).enumerate() {
        let bad = |got: String, want: String| FixtureError::Mismatch {
            what: "the layer map".into(),
            detail: format!("source layer {l} (fixture {f}) {got}, the kind keeps it {want}"),
        };
        let rope = |on: bool| if on { "roped" } else { "NoPE" };
        if src.rope_layer[l] != kind.rope {
            return Err(bad(
                format!("is {}", rope(src.rope_layer[l])),
                rope(kind.rope).into(),
            ));
        }
        let indexer = |on: bool| {
            if on {
                "a full indexer"
            } else {
                "a shared indexer"
            }
        };
        if src.indexer_full[l] != kind.full_indexer {
            return Err(bad(
                format!("has {}", indexer(src.indexer_full[l])),
                indexer(kind.full_indexer).into(),
            ));
        }
        let cache = layers[kind.cache];
        if src.kv_source[l] != cache {
            return Err(bad(
                format!("reads the cache of layer {}", src.kv_source[l]),
                format!("the cache of layer {cache}"),
            ));
        }
        let ffn = |dense: bool| if dense { "dense" } else { "routed" };
        let dense = l < src.dense_lead;
        if dense != kind.dense {
            return Err(bad(format!("is {}", ffn(dense)), ffn(kind.dense).into()));
        }
    }
    Ok(())
}

/// The hyperparameters of the source with the mini's widths, its layers still the source's own
/// numbers: what `Hparams::want` answers a source layer with in the fixture's dims.
fn resized(src: &Hparams) -> Hparams {
    Hparams {
        n_embd: HIDDEN as usize,
        n_head: HEADS as usize,
        dense_ff: DENSE_FF as usize,
        n_expert: EXPERTS as usize,
        expert_ff: FF as usize,
        indexer_top_k: TOP_K as usize,
        ..src.clone()
    }
}

/// The hyperparameters the fixture must read as: [`resized`], the per-layer arrays at the map's
/// layers and the cache map in fixture layers.
fn cut_of(src: &Hparams, layers: &[usize]) -> Hparams {
    let fixture_layer = |l: usize| {
        layers
            .iter()
            .position(|&m| m == l)
            .expect("check_map held every cache a map layer reads to the map")
    };
    Hparams {
        n_layer: layers.len(),
        rope_layer: layers.iter().map(|&l| src.rope_layer[l]).collect(),
        indexer_full: layers.iter().map(|&l| src.indexer_full[l]).collect(),
        kv_source: layers
            .iter()
            .map(|&l| fixture_layer(src.kv_source[l]))
            .collect(),
        ..resized(src)
    }
}

/// The mini geometry the plan writes every tensor at: [`resized`] of the source, asked layer by
/// layer. A name the reader does not hold is refused: every tensor of a file the reader accepts
/// is one of its stems, so the refusal names one the reader has not met.
struct Mini;

/// The tables of one plan: the source's hyperparameters at the mini's widths.
struct MiniTables(Hparams);

impl Tables for MiniTables {
    fn dims(
        &self,
        name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        let bad = |detail: String| FixtureError::Tensor {
            name: name.to_string(),
            detail,
        };
        if let Some(l) = layer {
            return self
                .0
                .want(l)
                .into_iter()
                .find(|(stem, _)| *stem == leaf)
                .map(|(_, d)| Some(d))
                .ok_or_else(|| {
                    bad("is no tensor of the mini table: the reader names every tensor of a mistral-large4 layer of this kind, so this one is new".into())
                });
        }
        if name == token_embd() || name == output() {
            let [_, vocab] = dims else {
                return Err(bad(format!(
                    "has dims {dims:?}; the globals are the hidden over the vocabulary"
                )));
            };
            return Ok(Some(vec![HIDDEN, *vocab]));
        }
        if name == output_norm() {
            return Ok(Some(vec![HIDDEN]));
        }
        // The rope's factors are the rope width's, which the mini keeps.
        if name == ROPE_FREQS {
            return Ok(None);
        }
        Err(bad(
            "is no tensor of the mini table: the reader names every tensor outside the layers of a mistral-large4 file, so this one is new".into(),
        ))
    }

    fn lines(&self) -> Vec<String> {
        vec![format!(
            "mistral-large4 mini: hidden {HIDDEN}, heads {HEADS} of {}, latent {}, rope {}, {} index heads of {}, \
             {EXPERTS} experts of width {FF}, dense lead {DENSE_FF}, index top-k {TOP_K}, the real vocabulary",
            self.0.head_k, self.0.kv_lora, self.0.rope_dims, self.0.indexer_heads, self.0.indexer_d
        )]
    }
}

impl Family for Mini {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// Every key a [`KeyRule::Set`] rewrites, which the plan only walks if the source has it.
    fn required_keys(&self) -> &'static [&'static str] {
        &[
            "embedding_length",
            "feed_forward_length",
            "attention.head_count",
            "expert_count",
            "expert_feed_forward_length",
            "attention.indexer.top_k",
        ]
    }

    /// The source must read as a `mistral-large4` file whose map layers are the kinds of
    /// [`KINDS`]; every dim of the mini is the reader's own for the source layer's kind.
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let src = Hparams::read(source)?;
        check_map(&src, &spec.layers)?;
        Ok(Box::new(MiniTables(resized(&src))))
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

/// `got` is `want`, or the mismatch of `key`.
fn same<T: PartialEq + std::fmt::Debug>(key: &str, got: &T, want: &T) -> Result<(), FixtureError> {
    if got == want {
        Ok(())
    } else {
        Err(FixtureError::Mismatch {
            what: key.to_string(),
            detail: format!("{got:?}, the cut's {want:?}"),
        })
    }
}

/// The engine reads `fixture`'s header as a `mistral-large4` file of the map's five layers
/// whose per-layer arrays are the source layers' own, the cache map in fixture layers, at the
/// mini's hidden, head count, experts, dense width and index list length, with the source's
/// latent, rope, head, index, router, YaRN, epsilon, softcap and vocabulary facts, and every
/// mapped tensor's type the source layer's — the (stem, type) pairs of the real file. Tensor
/// names and dims are the plan's, which `verify` holds the file to, and which the reader's own
/// dims check held the header to already.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, &spec.layers)?;
    let cut = cut_of(&src, &spec.layers);
    same("block_count", &fx.n_layer, &cut.n_layer)?;
    same("attention.rope_pattern", &fx.rope_layer, &cut.rope_layer)?;
    same(
        "attention.indexer.types",
        &fx.indexer_full,
        &cut.indexer_full,
    )?;
    same("attention.kv_source_map", &fx.kv_source, &cut.kv_source)?;
    same("embedding_length", &fx.n_embd, &cut.n_embd)?;
    same("attention.head_count", &fx.n_head, &cut.n_head)?;
    same("expert_count", &fx.n_expert, &cut.n_expert)?;
    same("expert_feed_forward_length", &fx.expert_ff, &cut.expert_ff)?;
    same("feed_forward_length", &fx.dense_ff, &cut.dense_ff)?;
    same(
        "attention.indexer.top_k",
        &fx.indexer_top_k,
        &cut.indexer_top_k,
    )?;
    same("context_length", &fx.n_ctx, &cut.n_ctx)?;
    same("vocab_size", &fx.n_vocab, &cut.n_vocab)?;
    same("attention.kv_lora_rank", &fx.kv_lora, &cut.kv_lora)?;
    same("attention.key_length", &fx.key_length, &cut.key_length)?;
    same("attention.key_length_mla", &fx.head_k, &cut.head_k)?;
    same("attention.value_length_mla", &fx.head_v, &cut.head_v)?;
    same("rope.dimension_count", &fx.rope_dims, &cut.rope_dims)?;
    same("rope.freq_base", &fx.rope_base, &cut.rope_base)?;
    same("rope.scaling", &fx.yarn, &cut.yarn)?;
    same(
        "attention.indexer.head_count",
        &fx.indexer_heads,
        &cut.indexer_heads,
    )?;
    same(
        "attention.indexer.key_length",
        &fx.indexer_d,
        &cut.indexer_d,
    )?;
    same(
        "attention.layer_norm_rms_epsilon",
        &fx.rms_eps,
        &cut.rms_eps,
    )?;
    same(
        "attention.layer_norm_epsilon",
        &fx.idx_norm_eps,
        &cut.idx_norm_eps,
    )?;
    same("leading_dense_block_count", &fx.dense_lead, &cut.dense_lead)?;
    same("expert_used_count", &fx.n_used, &cut.n_used)?;
    same("expert_shared_count", &fx.n_shared, &cut.n_shared)?;
    same("expert_weights_norm", &fx.weights_norm, &cut.weights_norm)?;
    same(
        "expert_weights_scale",
        &fx.weights_scale,
        &cut.weights_scale,
    )?;
    same("final_logit_softcapping", &fx.softcap, &cut.softcap)?;
    same("the keys read by default", &fx.defaults, &cut.defaults)?;
    for (f, &l) in spec.layers.iter().enumerate() {
        for (stem, _) in cut.want(f) {
            let (Some((_, a)), Some((_, b))) =
                (fixture.find(&blk(f, stem)), source.find(&blk(l, stem)))
            else {
                return Err(FixtureError::Mismatch {
                    what: format!("layer {f} (source {l})"),
                    detail: format!("{stem} is missing from the fixture or the source"),
                });
            };
            if a.ty != b.ty {
                return Err(FixtureError::Mismatch {
                    what: format!("layer {f} (source {l})"),
                    detail: format!("{stem} is {:?}, the source layer's {:?}", a.ty, b.ty),
                });
            }
        }
    }
    for global in [
        token_embd(),
        output_norm(),
        output(),
        ROPE_FREQS.to_string(),
    ] {
        let (Some((_, a)), Some((_, b))) = (fixture.find(&global), source.find(&global)) else {
            return Err(FixtureError::Mismatch {
                what: global,
                detail: "is missing from the fixture or the source".into(),
            });
        };
        if a.ty != b.ty {
            return Err(FixtureError::Mismatch {
                what: global,
                detail: format!("is {:?}, the source's {:?}", a.ty, b.ty),
            });
        }
    }
    Ok(fx)
}

#[cfg(test)]
mod tests {
    use super::super::names;
    use super::{
        DENSE_FF, EXPERTS, FF, HEADS, HIDDEN, INDEXER_K_GAIN, INDEXER_Q_GAIN, LEROPE_FACTOR,
        LN_BIAS_HALF_WIDTH, SSMAX_HALF_WIDTH, TOP_K, const_value, key_rule, spread_value,
    };
    use crate::arch::classify::ROPE_FREQS;
    use crate::fixture::KeyRule;

    /// The checked-in header table of the real shard (`tests/mistrallarge4_hdr.kv.json`).
    const KV: &str = include_str!("../../../tests/mistrallarge4_hdr.kv.json");

    /// Every `mistral-large4.*` key of the real header is covered, the six the mini resizes are
    /// the mini's values and the three per-layer arrays follow the map: the kv source map's
    /// values are layer ids, so it renumbers where the other two only select. Mutant: a rule
    /// removed (say `attention.indexer.top_k`) leaves the plan's `UncoveredKey` refusal to fire on
    /// the source's own key.
    #[test]
    fn every_source_key_is_covered() {
        let header: serde_json::Value = serde_json::from_str(KV).expect("the header table parses");
        let mut keys = 0;
        for key in header.as_object().expect("an object").keys() {
            let Some(suffix) = key.strip_prefix("mistral-large4.") else {
                continue;
            };
            keys += 1;
            assert!(key_rule(suffix).is_some(), "{suffix} is uncovered");
        }
        assert_eq!(keys, 39, "the real shard's architecture keys");
        for (key, want) in [
            ("embedding_length", HIDDEN),
            ("feed_forward_length", DENSE_FF),
            ("attention.head_count", HEADS),
            ("expert_count", EXPERTS),
            ("expert_feed_forward_length", FF),
            ("attention.indexer.top_k", TOP_K),
        ] {
            assert_eq!(key_rule(key), Some(KeyRule::Set(want)), "{key}");
        }
        for key in ["attention.rope_pattern", "attention.indexer.types"] {
            assert_eq!(key_rule(key), Some(KeyRule::PerLayer), "{key}");
        }
        assert_eq!(
            key_rule("attention.kv_source_map"),
            Some(KeyRule::PerLayerIds)
        );
        for key in ["expert_used_count", "attention.indexer.head_count"] {
            assert_eq!(key_rule(key), Some(KeyRule::Copy), "{key}");
        }
    }

    /// The fills: the norm gains 1, the indexer's two LayerNorm gains and the rope's factors
    /// non-unit constants, the LayerNorm biases and the SSMax pairs spreads, no leaf both, and
    /// the router's and the indexer's head-weight matrices no fill at all (they stay the
    /// generator's random matrices).
    #[test]
    fn the_one_dimensional_fills() {
        for (leaf, want) in [
            (names::ATTN_NORM, 1.0),
            (names::ATTN_KV_A_NORM, 1.0),
            (names::FFN_NORM, 1.0),
            ("output_norm.weight", 1.0),
            (names::INDEXER_Q_NORM_W, INDEXER_Q_GAIN),
            (names::INDEXER_K_NORM_W, INDEXER_K_GAIN),
            (ROPE_FREQS, LEROPE_FACTOR),
        ] {
            assert_eq!(
                const_value(leaf).map(f32::to_bits),
                Some(want.to_bits()),
                "{leaf}"
            );
        }
        for gain in [INDEXER_Q_GAIN, INDEXER_K_GAIN, LEROPE_FACTOR] {
            assert_ne!(gain, 1.0, "a unit value hides a body that skips it");
        }
        for (leaf, want) in [
            (names::INDEXER_Q_NORM_B, LN_BIAS_HALF_WIDTH),
            (names::INDEXER_K_NORM_B, LN_BIAS_HALF_WIDTH),
            (names::ATTN_SSMAX_W, SSMAX_HALF_WIDTH),
            (names::ATTN_SSMAX_B, SSMAX_HALF_WIDTH),
            (names::INDEXER_SSMAX_W, SSMAX_HALF_WIDTH),
            (names::INDEXER_SSMAX_B, SSMAX_HALF_WIDTH),
        ] {
            assert_eq!(spread_value(leaf), Some(want), "{leaf}");
            assert!(
                const_value(leaf).is_none(),
                "{leaf} is both a fill and a spread"
            );
        }
        for leaf in [names::FFN_GATE_INP, names::INDEXER_PROJ] {
            assert!(
                const_value(leaf).is_none() && spread_value(leaf).is_none(),
                "{leaf} stays a matrix"
            );
        }
    }
}
