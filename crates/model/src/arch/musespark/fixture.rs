//! The Muse Spark Open (`muse-spark`) gate fixture's spec ([`spec`]): the facts of a
//! `muse-spark` file the common generator (`crate::fixture`) cannot know. The model is one
//! repeated layer — grouped-query attention over a sigmoid-routed mixture of SiLU-gated
//! experts in a latent, every layer routing, no dense layer — and the file is a six-shard
//! split set.
//!
//! The fixture is a requantized mini-Muse, not a cut of the real layers' shapes: hidden 1,024,
//! 16 query heads over 2 KV heads of the real head 64 (the same group 8), latent 512, 16
//! experts of width 256 — every dimension the family's [`Tables`] write, every header key
//! that names one a [`KeyRule::Set`]. Every layer is a source layer, renamed `blk.L.` →
//! `blk.f.`, and the map is the first four, whose window-pattern entries are `[T,F,T,T]`:
//!
//! | fixture layer | source layer | the pattern | why |
//! |---|---|---|---|
//! | f0 | L0 | sliding | the first sliding layer: rope at the sliding base inside its window |
//! | f1 | L1 | full | the first full layer: NoPE, no window — the T→F transition |
//! | f2 | L2 | sliding | the F→T transition |
//! | f3 | L3 | sliding | the T→T run |
//!
//! What it keeps: the pattern `[T,F,T,T]` (both window kinds, both transitions and the run
//! between them — a "3:1" rule alone cannot decide the real file's 62-layer tail), every
//! (tensor stem, type) pair of the real file's layers — IQ2_XXS routed experts, Q6_K
//! `attn_q`/`attn_output`/`output`, Q8_0 `attn_k`/`attn_v`/`attn_gate` and the latent pair,
//! Q4_K `token_embd`, F32 norms/router/α/β/bias — and the real vocabulary: the globals keep
//! their second axis, and the tokenizer tables are the source's copied ones, so the fixture's
//! tokenizer is the file's. The router's `expert_used_count` stays 8, the real top-k, over
//! the 16 rivals of the mini field, where the selection-only bias and the unbiased-weight
//! renormalisation both still decide picks ([`BIAS_HALF_WIDTH`]). What it minifies: the
//! hidden, the head counts, the latent, the expert count and the expert width.
//!
//! The router's weights are pre-scaled by the MetaP output multiplier at conversion and the
//! reader multiplies nothing on top; the generator draws `ffn_gate_inp` at the matrix band
//! every matrix takes (σ 1/√K) and adds no factor, so the bytes hold the pre-scaled
//! convention and a body that multiplies again diverges from the oracle at the router's
//! logits by a constant per logit.
//!
//! The tensor data is 309,895,936 bytes in 95 tensors [derived from the header's tensor sizes
//! at the mini geometry]: 286,104,064 of it the three globals, 23,791,872 the four layers. It
//! is one shard ([`SHARD_BYTES`]): the body loads the whole file on the card, and that loader
//! reads one shard; the split set comes back with the placed loader.
//!
//! The card budget is fixed, not planned: the architecture has no program yet (`program_of`
//! is `None`), so no plan of the written file chooses a cap; the recorded 2 GiB keeps the
//! whole fixture resident, and the engine's own gate overrides it. No draft file (the real
//! file holds no next-token tensors) and no r8 sidecar.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{
    ATTN_GATE, ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_RESID_ALPHA,
    ATTN_RESID_BETA, ATTN_V, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_GATE_EXPS, FFN_GATE_INP,
    FFN_LATENT_DOWN, FFN_LATENT_UP, FFN_NORM, FFN_POST_EXP_NORM, FFN_PRE_EXP_NORM, FFN_RESID_ALPHA,
    FFN_RESID_BETA, FFN_UP_EXPS, POST_FFW_NORM,
};
use crate::arch::names::{output, output_norm, token_embd};
use crate::fixture::{CardBudget, Family, FixtureError, FixtureSpec, KeyRule, Tables, Window};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above): the first four, the
/// pattern entries the fixture keeps.
pub const LAYER_MAP: [usize; 4] = [0, 1, 2, 3];

/// The hidden the mini writes: `embedding_length`, the query field and the residual stream.
pub const HIDDEN: u64 = 1024;
/// The query heads the mini writes.
pub const HEADS: u64 = 16;
/// The KV heads the mini writes: with [`HEADS`], the real group 8.
pub const KV_HEADS: u64 = 2;
/// The latent the mini writes: `moe_latent_size`, the experts' own width.
pub const LATENT: u64 = 512;
/// The experts the mini writes: `expert_count`.
pub const EXPERTS: u64 = 16;
/// One expert's width the mini writes: `expert_feed_forward_length`, whole IQ2_XXS blocks.
pub const FF: u64 = 256;

/// The d window, [2^-14, 2^-4]: normal f16 values every type's scale fits at every K the
/// mini holds. The widest is the IQ2_XXS down stack at K = 256, whose σ 1/16 needs d about
/// 2^-5 [derived: σ over the grid's largest magnitude, ≈ 2], so the top is 2^-4; the floor is
/// the smallest normal f16.
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 16.0;

/// The most tensor data a shard holds: the shared default, above the fixture's 310 MB, so the
/// fixture is one shard (the whole-card loader, `GpuModel::load_blocks`, reads one).
pub const SHARD_BYTES: u64 = crate::fixture::DEFAULT_SHARD_BYTES;

/// The card byte budget the fixture records. Above the fixture's tensor data (296 MB), it
/// keeps the whole model resident; the architecture has no program to plan a cap through, and
/// the engine's attach sets the value its placement reads.
pub const CARD_BUDGET: u64 = 2 << 30;

/// The target file is `<STEM>-00001-of-00003.gguf`.
pub const STEM: &str = "musespark-fixture";

/// The real first shard `verify` opens when none is named: the profile's
/// (`tools/ref/models/muse-spark.sh`).
pub const DEFAULT_MODEL: &str = "/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf";

const ARCH: &str = "muse-spark";

/// The gain the mini writes into `attn_q_norm`: the real file holds the MetaP
/// `qk_scale_factor` as ones scaled by it (a constant per head, its value the checkpoint's
/// own, not a header key). The mini's constant is 2, not 1, so a body that skips the gain
/// still reads the oracle wrong; `attn_k_norm` stays 1, the conversion's ones.
pub const QK_GAIN: f32 = 2.0;

/// The half-width of the router's selection bias (`exp_probs_b`), drawn per expert. A constant
/// bias moves no pick, so a body that skipped the add would pass. The router's scores are
/// sigmoid(logits) of unit-variance logits over a unit-RMS row, spread over about (0.2, 0.8),
/// and the gap between the 8th and 9th of 16 such scores is about 0.04 [derived: the uniform
/// order-statistic spacing over that range], so a uniform ±0.05 moves picks at the top-8 line
/// and fixes none.
pub const BIAS_HALF_WIDTH: f32 = 0.05;

/// The half-width of the residual mix vectors (`attn_resid_{alpha,beta}`,
/// `ffn_resid_{alpha,beta}`), drawn per channel. The real vectors are per-channel by form —
/// the converter resolves both from one gate weight — and a constant pair would let a body
/// that broadcasts one value pass; ±0.05 keeps the mix's output near zero mean (the next
/// sublayer's norm restores the scale) while every channel reads its own α and β.
pub const RESID_HALF_WIDTH: f32 = 0.05;

/// The Muse Spark fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the profile's
/// file as the default source, no ff override (the mini's expert width is a header key, not
/// an ff axis), no draft and a fixed card budget.
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

/// The coverage table of the architecture keys; `None` is a key the map does not cover. What
/// the mini resizes is a [`KeyRule::Set`]; the pattern array follows the map; the rest — the
/// epsilons, the sliding window and base, the logit scale and softcap, the router's gating
/// func, weight norm and scale, the top-k, the head width and the context — is copied, and so
/// are the two keys the graph does not read (`feed_forward_length`, `rope.freq_base`: the
/// graph takes the sliding base).
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 16] = [
        "context_length",
        "feed_forward_length",
        "vocab_size",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "attention.sliding_window",
        "rope.freq_base",
        "rope.freq_base_swa",
        "post_norm_epsilon",
        "logit_scale",
        "final_logit_softcapping",
        "expert_used_count",
        "expert_gating_func",
        "expert_weights_norm",
        "expert_weights_scale",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.sliding_window_pattern" => KeyRule::PerLayer,
        "embedding_length" => KeyRule::Set(HIDDEN),
        "attention.head_count" => KeyRule::Set(HEADS),
        "attention.head_count_kv" => KeyRule::Set(KV_HEADS),
        "moe_latent_size" => KeyRule::Set(LATENT),
        "expert_count" => KeyRule::Set(EXPERTS),
        "expert_feed_forward_length" => KeyRule::Set(FF),
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the norm gains 1,
/// among them the k norm's (the conversion's ones), and the q norm's MetaP gain
/// ([`QK_GAIN`]). The residual mix vectors and the router's selection bias are spreads
/// ([`spread_value`]).
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 8] = [
        (ATTN_NORM, 1.0),
        (ATTN_K_NORM, 1.0),
        (ATTN_Q_NORM, QK_GAIN),
        (FFN_NORM, 1.0),
        (FFN_PRE_EXP_NORM, 1.0),
        (FFN_POST_EXP_NORM, 1.0),
        (POST_FFW_NORM, 1.0),
        // The global's leaf, spelled as the common name helper does (`output_norm.weight`):
        // the globals' names are its functions, not constants.
        ("output_norm.weight", 1.0),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The spread of an F32 tensor the family names: the residual mix vectors
/// ([`RESID_HALF_WIDTH`]) and the router's selection bias ([`BIAS_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    match leaf {
        ATTN_RESID_ALPHA | ATTN_RESID_BETA | FFN_RESID_ALPHA | FFN_RESID_BETA => {
            Some(RESID_HALF_WIDTH)
        }
        EXP_PROBS_B => Some(BIAS_HALF_WIDTH),
        _ => None,
    }
}

/// The pattern the map must read from the source's own array, layer by layer: sliding, full,
/// sliding, sliding — the real file's first four entries, never reconstructed from a rule.
fn check_pattern(hp: &Hparams, layers: &[usize]) -> Result<(), FixtureError> {
    const WANT: [bool; 4] = [true, false, true, true];
    if layers.len() != WANT.len() {
        return Err(FixtureError::Mismatch {
            what: "the layer map".into(),
            detail: format!(
                "holds {} layers, the fixture is built for the four of the pattern [sliding, \
                 full, sliding, sliding]",
                layers.len()
            ),
        });
    }
    for (f, (&l, want)) in layers.iter().zip(WANT).enumerate() {
        let got = hp.sliding[l];
        if got != want {
            return Err(FixtureError::Mismatch {
                what: "the layer map".into(),
                detail: format!(
                    "source layer {l} (fixture {f}) is {}, the pattern keeps it {}",
                    if got { "sliding" } else { "full" },
                    if want { "sliding" } else { "full" },
                ),
            });
        }
    }
    Ok(())
}

/// The mini geometry the plan writes every tensor at. A name the table does not hold is
/// refused: every tensor of a file the reader accepts is one of these, so the refusal names a
/// stem the reader has not met.
struct Mini;

impl Mini {
    /// The mini dims of layer stem `leaf`, `None` for one the table does not hold. The head
    /// width stays the real file's 64: the rope width and the q/k norms read it.
    fn stem_dims(leaf: &str) -> Option<Vec<u64>> {
        let e = HIDDEN;
        let d = 64;
        let (q, kv) = (HEADS * d, KV_HEADS * d);
        let (x, w, lat) = (EXPERTS, FF, LATENT);
        let dims = match leaf {
            ATTN_NORM => vec![e],
            ATTN_Q => vec![e, q],
            ATTN_K => vec![e, kv],
            ATTN_V => vec![e, kv],
            ATTN_Q_NORM => vec![d],
            ATTN_K_NORM => vec![d],
            ATTN_GATE => vec![e, q],
            ATTN_OUTPUT => vec![q, e],
            ATTN_RESID_ALPHA => vec![e],
            ATTN_RESID_BETA => vec![e],
            FFN_NORM => vec![e],
            FFN_LATENT_DOWN => vec![e, lat],
            FFN_PRE_EXP_NORM => vec![lat],
            FFN_GATE_INP => vec![e, x],
            EXP_PROBS_B => vec![x],
            FFN_GATE_EXPS => vec![lat, w, x],
            FFN_UP_EXPS => vec![lat, w, x],
            FFN_DOWN_EXPS => vec![w, lat, x],
            FFN_POST_EXP_NORM => vec![lat],
            FFN_LATENT_UP => vec![lat, e],
            POST_FFW_NORM => vec![e],
            FFN_RESID_ALPHA => vec![e],
            FFN_RESID_BETA => vec![e],
            _ => return None,
        };
        Some(dims)
    }
}

impl Tables for Mini {
    fn dims(
        &self,
        name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        let _ = layer;
        if let Some(d) = Mini::stem_dims(leaf) {
            return Ok(Some(d));
        }
        let bad = |detail: String| FixtureError::Tensor {
            name: name.to_string(),
            detail,
        };
        if leaf == token_embd().as_str() || leaf == output().as_str() {
            let [_, vocab] = dims else {
                return Err(bad(format!(
                    "has dims {dims:?}; the globals are the hidden over the vocabulary"
                )));
            };
            if *vocab == 0 {
                return Err(bad("holds no vocabulary".into()));
            }
            return Ok(Some(vec![HIDDEN, *vocab]));
        }
        if leaf == output_norm().as_str() {
            if dims.first().is_none_or(|&n| n == 0) {
                return Err(bad("holds no value".into()));
            }
            return Ok(Some(vec![HIDDEN]));
        }
        Err(bad(
            "is no tensor of the mini table: the reader names every tensor of a muse-spark \
             file, so this one is new"
                .into(),
        ))
    }

    fn lines(&self) -> Vec<String> {
        vec![format!(
            "musespark mini: hidden {HIDDEN}, heads {HEADS}/{KV_HEADS} of 64, latent {LATENT}, \
             {EXPERTS} experts of width {FF}, top-8 of the real file, the real vocabulary"
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
            "attention.head_count",
            "attention.head_count_kv",
            "moe_latent_size",
            "expert_count",
            "expert_feed_forward_length",
        ]
    }

    /// The source must read as a `muse-spark` file whose map layers spell the pattern
    /// `[T,F,T,T]`; the mini dims are the family's constants, so nothing else of the source
    /// moves with the map.
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let hp = Hparams::read(source)?;
        check_pattern(&hp, &spec.layers)?;
        Ok(Box::new(Mini))
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

/// The engine reads `fixture`'s header as a `muse-spark` file of the map's four layers whose
/// window pattern is the source layers' own `[T,F,T,T]`, at the mini hidden, heads, latent
/// and expert field, with the source's head width, top-k, vocabulary, epsilons, window, rope
/// base, logit scale, softcap and router constants, and every mapped tensor's type the source
/// layer's — the (stem, type) pairs of the real file. Tensor names and dims are the plan's,
/// which `verify` holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_pattern(&src, &spec.layers)?;
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    if fx.n_layer != spec.layers.len() {
        return Err(bad(
            "layer count".into(),
            format!("{}, not {}", fx.n_layer, spec.layers.len()),
        ));
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        if fx.sliding[f] != src.sliding[l] {
            return Err(bad(
                format!("layer {f} (source {l})"),
                format!(
                    "{}, the source layer's {}",
                    if fx.sliding[f] { "sliding" } else { "full" },
                    if src.sliding[l] { "sliding" } else { "full" },
                ),
            ));
        }
    }
    for (key, a, b) in [
        ("embedding_length", fx.n_embd as u64, HIDDEN),
        ("attention.head_count", fx.n_head as u64, HEADS),
        ("attention.head_count_kv", fx.n_head_kv as u64, KV_HEADS),
        ("moe_latent_size", fx.latent as u64, LATENT),
        ("expert_count", fx.n_expert as u64, EXPERTS),
        (
            "expert_feed_forward_length",
            fx.expert_ff.first().copied().unwrap_or(0) as u64,
            FF,
        ),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, the mini writes {b}")));
        }
    }
    for (key, a, b) in [
        ("attention.key_length", fx.head_dim, src.head_dim),
        ("vocab_size", fx.n_vocab, src.n_vocab),
        ("expert_used_count", fx.n_used, src.n_used),
        ("attention.sliding_window", fx.window, src.window),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, the source's {b}")));
        }
    }
    for (key, a, b) in [
        ("attention.layer_norm_rms_epsilon", fx.rms_eps, src.rms_eps),
        ("post_norm_epsilon", fx.post_eps, src.post_eps),
        ("rope.freq_base_swa", fx.rope_base, src.rope_base),
        ("logit_scale", fx.logit_scale, src.logit_scale),
        ("final_logit_softcapping", fx.softcap, src.softcap),
        ("expert_weights_scale", fx.weights_scale, src.weights_scale),
    ] {
        if a.to_bits() != b.to_bits() {
            return Err(bad(key.into(), format!("{a}, the source's {b}")));
        }
    }
    if fx.weights_norm != src.weights_norm || fx.defaults != src.defaults {
        return Err(bad(
            "the router and the keys read by default".into(),
            format!(
                "weights_norm {}, the source's {}; defaults {:?}, the source's {:?}",
                fx.weights_norm, src.weights_norm, fx.defaults, src.defaults
            ),
        ));
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        for stem in STEMS {
            let (Some((_, a)), Some((_, b))) = (
                fixture.find(&crate::arch::names::blk(f, stem)),
                source.find(&crate::arch::names::blk(l, stem)),
            ) else {
                return Err(bad(
                    format!("layer {f} (source {l})"),
                    format!("{stem} is missing from the fixture or the source"),
                ));
            };
            if a.ty != b.ty {
                return Err(bad(
                    format!("layer {f} (source {l})"),
                    format!("{stem} is {:?}, the source layer's {:?}", a.ty, b.ty),
                ));
            }
        }
    }
    for global in [token_embd(), output_norm(), output()] {
        let (Some((_, a)), Some((_, b))) = (fixture.find(&global), source.find(&global)) else {
            return Err(bad(
                global,
                "is missing from the fixture or the source".into(),
            ));
        };
        if a.ty != b.ty {
            return Err(bad(
                global,
                format!("is {:?}, the source's {:?}", a.ty, b.ty),
            ));
        }
    }
    Ok(fx)
}

/// Every layer stem, in the file's own order.
const STEMS: &[&str] = &[
    ATTN_NORM,
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_Q_NORM,
    ATTN_K_NORM,
    ATTN_GATE,
    ATTN_OUTPUT,
    ATTN_RESID_ALPHA,
    ATTN_RESID_BETA,
    FFN_NORM,
    FFN_LATENT_DOWN,
    FFN_PRE_EXP_NORM,
    FFN_GATE_INP,
    EXP_PROBS_B,
    FFN_GATE_EXPS,
    FFN_UP_EXPS,
    FFN_DOWN_EXPS,
    FFN_POST_EXP_NORM,
    FFN_LATENT_UP,
    POST_FFW_NORM,
    FFN_RESID_ALPHA,
    FFN_RESID_BETA,
];

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::super::names;
    use super::{
        EXP_PROBS_B, EXPERTS, FF, HIDDEN, KV_HEADS, LATENT, Mini, QK_GAIN, RESID_HALF_WIDTH,
        check_kinds, const_value, key_rule, spec, spread_value,
    };
    use crate::arch::names::{blk, output, output_norm, token_embd};
    use crate::arch::synthetic::{Ty, V, header_typed};
    use crate::fixture::{Family, KeyRule, Tables};

    /// The fixture's key/value rows, `mutate`'s subject.
    type MiniKv = Vec<(&'static str, V)>;
    /// The fixture's tensor rows — name, dims, type — `retype`'s subject.
    type MiniTensors = Vec<(String, Vec<u64>, Ty)>;

    /// A readable `muse-spark` source of five layers, the pattern `[T,F,T,T,F]` — the first
    /// four the map keeps — at a tiny geometry with the real file's head width, top-k and
    /// per-stem types: the mini dims are the family's constants, so the source's own shapes do
    /// not enter the plan, but the facts the fixture copies (the head, the top-k) match it.
    const SLIDING: [bool; 5] = [true, false, true, true, false];
    // The source's row widths are whole 256-value blocks of its K and IQ types.
    const EMBD: u64 = 256;
    const HEADS: u32 = 4;
    const KV: u32 = 2;
    const HEAD: u64 = 64;
    const EXPERTS_SRC: u64 = 8;
    const FF_SRC: u64 = 256;
    const LATENT_SRC: u64 = 256;

    /// One layer's stems with their types, at `heads` query heads and latent `latent`.
    fn layer_tensors(l: usize, heads: u32, latent: u64, out: &mut MiniTensors) {
        let e = EMBD;
        let (q, kv) = (heads as u64 * HEAD, KV as u64 * HEAD);
        let (x, w, lat) = (EXPERTS_SRC, FF_SRC, latent);
        for (stem, dims, ty) in [
            (names::ATTN_NORM, vec![e], Ty::F32),
            (names::ATTN_Q, vec![e, q], Ty::Q6K),
            (names::ATTN_K, vec![e, kv], Ty::Q8_0),
            (names::ATTN_V, vec![e, kv], Ty::Q8_0),
            (names::ATTN_Q_NORM, vec![HEAD], Ty::F32),
            (names::ATTN_K_NORM, vec![HEAD], Ty::F32),
            (names::ATTN_GATE, vec![e, q], Ty::Q8_0),
            (names::ATTN_OUTPUT, vec![q, e], Ty::Q6K),
            (names::ATTN_RESID_ALPHA, vec![e], Ty::F32),
            (names::ATTN_RESID_BETA, vec![e], Ty::F32),
            (names::FFN_NORM, vec![e], Ty::F32),
            (names::FFN_LATENT_DOWN, vec![e, lat], Ty::Q8_0),
            (names::FFN_PRE_EXP_NORM, vec![lat], Ty::F32),
            (names::FFN_GATE_INP, vec![e, x], Ty::F32),
            (names::EXP_PROBS_B, vec![x], Ty::F32),
            (names::FFN_GATE_EXPS, vec![lat, w, x], Ty::Iq2Xxs),
            (names::FFN_UP_EXPS, vec![lat, w, x], Ty::Iq2Xxs),
            (names::FFN_DOWN_EXPS, vec![w, lat, x], Ty::Iq2Xxs),
            (names::FFN_POST_EXP_NORM, vec![lat], Ty::F32),
            (names::FFN_LATENT_UP, vec![lat, e], Ty::Q8_0),
            (names::POST_FFW_NORM, vec![e], Ty::F32),
            (names::FFN_RESID_ALPHA, vec![e], Ty::F32),
            (names::FFN_RESID_BETA, vec![e], Ty::F32),
        ] {
            out.push((blk(l, stem), dims, ty));
        }
    }

    fn keys(n_layer: usize, sliding: &[bool]) -> MiniKv {
        vec![
            ("block_count", V::U32(n_layer as u32)),
            ("embedding_length", V::U32(EMBD as u32)),
            ("context_length", V::U32(128)),
            ("attention.head_count", V::U32(HEADS)),
            ("attention.head_count_kv", V::U32(KV)),
            ("attention.key_length", V::U32(HEAD as u32)),
            ("attention.value_length", V::U32(HEAD as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.sliding_window", V::U32(16)),
            (
                "attention.sliding_window_pattern",
                V::Bools(sliding.to_vec()),
            ),
            ("rope.freq_base_swa", V::F32(5e5)),
            ("post_norm_epsilon", V::F32(1e-8)),
            ("logit_scale", V::F32(0.1767767)),
            ("final_logit_softcapping", V::F32(20.0)),
            ("expert_count", V::U32(EXPERTS_SRC as u32)),
            ("expert_used_count", V::U32(8)),
            ("expert_feed_forward_length", V::U32(FF_SRC as u32)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_norm", V::Bool(true)),
            ("expert_weights_scale", V::F32(1.0)),
            ("moe_latent_size", V::U32(LATENT_SRC as u32)),
        ]
    }

    const CHAT: [(&str, V); 2] = [
        ("tokenizer.ggml.pre", V::Str("gpt-4o")),
        ("tokenizer.chat_template", V::Str("{{ bos }}{{ messages }}")),
    ];

    fn source_tensors() -> MiniTensors {
        let mut t = vec![
            (token_embd(), vec![EMBD, 2], Ty::Q4K),
            (output_norm(), vec![EMBD], Ty::F32),
            (output(), vec![EMBD, 2], Ty::Q6K),
        ];
        for l in 0..SLIDING.len() {
            layer_tensors(l, HEADS, LATENT_SRC, &mut t);
        }
        t
    }

    fn source(tag: &str, sliding: &[bool]) -> std::path::PathBuf {
        header_typed(
            tag,
            "muse-spark",
            &keys(SLIDING.len(), sliding),
            &CHAT,
            &source_tensors(),
        )
    }

    /// The header of the fixture the plan writes: the mini keys and dims, the source's copied
    /// facts, every tensor at the mini geometry with the real file's types. `heads` and
    /// `latent` resize their keys and tensors together (a key alone would fail the reader's
    /// own dims check before the family's), `mutate` rewrites one key, `retype` one type.
    fn fixture(
        tag: &str,
        heads: u32,
        latent: u64,
        mutate: &dyn Fn(&mut MiniKv),
        retype: &dyn Fn(&mut MiniTensors),
    ) -> std::path::PathBuf {
        let mut kv: MiniKv = vec![
            ("block_count", V::U32(4)),
            ("embedding_length", V::U32(HIDDEN as u32)),
            ("context_length", V::U32(128)),
            ("attention.head_count", V::U32(heads)),
            ("attention.head_count_kv", V::U32(KV_HEADS as u32)),
            ("attention.key_length", V::U32(64)),
            ("attention.value_length", V::U32(64)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.sliding_window", V::U32(16)),
            (
                "attention.sliding_window_pattern",
                V::Bools(vec![true, false, true, true]),
            ),
            ("rope.freq_base_swa", V::F32(5e5)),
            ("post_norm_epsilon", V::F32(1e-8)),
            ("logit_scale", V::F32(0.1767767)),
            ("final_logit_softcapping", V::F32(20.0)),
            ("expert_count", V::U32(EXPERTS as u32)),
            ("expert_used_count", V::U32(8)),
            ("expert_feed_forward_length", V::U32(FF as u32)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_norm", V::Bool(true)),
            ("expert_weights_scale", V::F32(1.0)),
            ("moe_latent_size", V::U32(latent as u32)),
        ];
        mutate(&mut kv);
        let e = HIDDEN;
        let (q, kvd) = (heads as u64 * 64, KV_HEADS * 64);
        let (x, w, lat) = (EXPERTS, FF, latent);
        let mut t: MiniTensors = vec![
            (token_embd(), vec![e, 2], Ty::Q4K),
            (output_norm(), vec![e], Ty::F32),
            (output(), vec![e, 2], Ty::Q6K),
        ];
        for l in 0..4 {
            for (stem, dims, ty) in [
                (names::ATTN_NORM, vec![e], Ty::F32),
                (names::ATTN_Q, vec![e, q], Ty::Q6K),
                (names::ATTN_K, vec![e, kvd], Ty::Q8_0),
                (names::ATTN_V, vec![e, kvd], Ty::Q8_0),
                (names::ATTN_Q_NORM, vec![64], Ty::F32),
                (names::ATTN_K_NORM, vec![64], Ty::F32),
                (names::ATTN_GATE, vec![e, q], Ty::Q8_0),
                (names::ATTN_OUTPUT, vec![q, e], Ty::Q6K),
                (names::ATTN_RESID_ALPHA, vec![e], Ty::F32),
                (names::ATTN_RESID_BETA, vec![e], Ty::F32),
                (names::FFN_NORM, vec![e], Ty::F32),
                (names::FFN_LATENT_DOWN, vec![e, lat], Ty::Q8_0),
                (names::FFN_PRE_EXP_NORM, vec![lat], Ty::F32),
                (names::FFN_GATE_INP, vec![e, x], Ty::F32),
                (names::EXP_PROBS_B, vec![x], Ty::F32),
                (names::FFN_GATE_EXPS, vec![lat, w, x], Ty::Iq2Xxs),
                (names::FFN_UP_EXPS, vec![lat, w, x], Ty::Iq2Xxs),
                (names::FFN_DOWN_EXPS, vec![w, lat, x], Ty::Iq2Xxs),
                (names::FFN_POST_EXP_NORM, vec![lat], Ty::F32),
                (names::FFN_LATENT_UP, vec![lat, e], Ty::Q8_0),
                (names::POST_FFW_NORM, vec![e], Ty::F32),
                (names::FFN_RESID_ALPHA, vec![e], Ty::F32),
                (names::FFN_RESID_BETA, vec![e], Ty::F32),
            ] {
                t.push((blk(l, stem), dims, ty));
            }
        }
        retype(&mut t);
        header_typed(tag, "muse-spark", &kv, &CHAT, &t)
    }

    fn set(kv: &mut [(&'static str, V)], key: &str, v: V) {
        kv.iter_mut().find(|(k, _)| *k == key).expect(key).1 = v;
    }

    fn open(path: &std::path::Path) -> gguf::Split {
        gguf::Split::open(path).expect("the synthetic header opens")
    }

    /// The map reads the pattern from the source's own array: a source whose layer 1 slides
    /// is refused by name, and a map of another length than four too.
    #[test]
    fn the_map_keeps_the_pattern_read_not_reconstructed() {
        let path = source("muse-fx-pattern-ok", &SLIDING);
        let split = open(&path);
        Mini.tables(&split, &spec())
            .expect("the first four layers spell [T,F,T,T]");
        drop(split);
        let _ = std::fs::remove_file(&path);

        let mut sliding = SLIDING;
        sliding[1] = true;
        let path = source("muse-fx-pattern-flip", &sliding);
        let split = open(&path);
        let err = Mini
            .tables(&split, &spec())
            .err()
            .expect("the map's layer 1 is sliding");
        assert!(
            err.to_string()
                .contains("source layer 1 (fixture 1) is sliding"),
            "{err}"
        );
        drop(split);
        let _ = std::fs::remove_file(&path);

        let mut short = spec();
        short.layers = vec![0, 1];
        let path = source("muse-fx-pattern-short", &SLIDING);
        let split = open(&path);
        let err = Mini
            .tables(&split, &short)
            .err()
            .expect("a map of two layers");
        assert!(
            err.to_string()
                .contains("holds 2 layers, the fixture is built for the four"),
            "{err}"
        );
        drop(split);
        let _ = std::fs::remove_file(&path);
    }

    /// Every layer stem takes the mini dims, the head width the real file's; the globals keep
    /// the source's vocabulary axis; a name the table does not hold is refused by name.
    #[test]
    fn every_tensor_takes_the_mini_dims() {
        for (leaf, want) in [
            (names::ATTN_Q, vec![HIDDEN, super::HEADS * 64]),
            (names::ATTN_K, vec![HIDDEN, KV_HEADS * 64]),
            (names::ATTN_Q_NORM, vec![64]),
            (names::FFN_GATE_EXPS, vec![LATENT, FF, EXPERTS]),
            (names::FFN_DOWN_EXPS, vec![FF, LATENT, EXPERTS]),
            (names::FFN_GATE_INP, vec![HIDDEN, EXPERTS]),
        ] {
            assert_eq!(
                Mini.dims(&format!("blk.0.{leaf}"), Some(0), leaf, &[8192, 8192])
                    .expect("the stem is in the table"),
                Some(want),
                "{leaf}"
            );
        }
        assert_eq!(
            Mini.dims(&token_embd(), None, "token_embd.weight", &[8192, 202048])
                .expect("the embedding is in the table"),
            Some(vec![HIDDEN, 202048]),
            "the embedding keeps the vocabulary axis"
        );
        assert_eq!(
            Mini.dims(&output(), None, "output.weight", &[8192, 202048])
                .expect("the head is in the table"),
            Some(vec![HIDDEN, 202048])
        );
        assert_eq!(
            Mini.dims(&output_norm(), None, "output_norm.weight", &[8192])
                .expect("the norm is in the table"),
            Some(vec![HIDDEN])
        );
        let err = Mini
            .dims(
                "blk.0.ffn_down_shexp.weight",
                Some(0),
                "ffn_down_shexp.weight",
                &[8192, 4096],
            )
            .expect_err("a stem the reader does not hold");
        assert!(
            err.to_string().contains("is no tensor of the mini table"),
            "{err}"
        );
    }

    /// Every key of the real header — the 23 `muse-spark.*` keys the specimen carries, and
    /// `vocab_size` for a file that spells it — is covered, and the six the mini resizes are
    /// the mini's values. Mutant: a rule removed (say `moe_latent_size`) leaves the plan's
    /// `UncoveredKey` refusal to fire on the source's own key.
    #[test]
    fn every_source_key_is_covered() {
        for key in [
            "block_count",
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.layer_norm_rms_epsilon",
            "attention.sliding_window",
            "attention.sliding_window_pattern",
            "rope.freq_base",
            "rope.freq_base_swa",
            "post_norm_epsilon",
            "logit_scale",
            "final_logit_softcapping",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_gating_func",
            "expert_weights_norm",
            "expert_weights_scale",
            "moe_latent_size",
            "vocab_size",
        ] {
            assert!(key_rule(key).is_some(), "{key} is uncovered");
        }
        for (key, want) in [
            ("embedding_length", HIDDEN),
            ("attention.head_count", super::HEADS),
            ("attention.head_count_kv", KV_HEADS),
            ("moe_latent_size", LATENT),
            ("expert_count", EXPERTS),
            ("expert_feed_forward_length", FF),
        ] {
            assert_eq!(key_rule(key), Some(KeyRule::Set(want)), "{key}");
        }
        assert_eq!(
            key_rule("attention.sliding_window_pattern"),
            Some(KeyRule::PerLayer)
        );
        assert_eq!(key_rule("expert_used_count"), Some(KeyRule::Copy));
    }

    /// The fills: the norm gains 1, the q norm's gain [`QK_GAIN`] beside the k norm's 1, the
    /// residual mix vectors and the router's bias spreads, no leaf both a constant and a
    /// spread, and the router's weight matrix no fill at all (it stays the generator's
    /// random matrix, the pre-scaled bytes).
    #[test]
    fn the_one_dimensional_fills() {
        for (leaf, want) in [
            (names::ATTN_NORM, 1.0),
            (names::ATTN_K_NORM, 1.0),
            (names::ATTN_Q_NORM, QK_GAIN),
            (names::FFN_NORM, 1.0),
            (names::FFN_PRE_EXP_NORM, 1.0),
            (names::FFN_POST_EXP_NORM, 1.0),
            (names::POST_FFW_NORM, 1.0),
            ("output_norm.weight", 1.0),
        ] {
            assert_eq!(
                const_value(leaf).map(f32::to_bits),
                Some(want.to_bits()),
                "{leaf}"
            );
        }
        assert_ne!(QK_GAIN, 1.0, "a unit gain hides a body that skips it");
        for leaf in [
            names::ATTN_RESID_ALPHA,
            names::ATTN_RESID_BETA,
            names::FFN_RESID_ALPHA,
            names::FFN_RESID_BETA,
            EXP_PROBS_B,
        ] {
            assert_eq!(spread_value(leaf), Some(RESID_HALF_WIDTH), "{leaf}");
            assert!(
                const_value(leaf).is_none(),
                "{leaf} is both a fill and a spread"
            );
        }
        assert!(
            spread_value(names::FFN_GATE_INP).is_none(),
            "the router stays a matrix"
        );
    }

    /// The fixture reads as the mini: its own pattern, the mini's resized keys, the source's
    /// copied facts and every mapped tensor's type the source layer's. Each mutant is refused
    /// by name: a pattern entry flipped, a resized key left at the source's value, a copied
    /// key changed, a tensor of another type, a type change in a global.
    #[test]
    fn the_fixture_reads_as_the_mini_and_the_source() {
        let src = source("muse-fx-src", &SLIDING);
        let src_split = open(&src);
        let check = |tag: &str,
                     heads: u32,
                     latent: u64,
                     mutate: &dyn Fn(&mut MiniKv),
                     retype: &dyn Fn(&mut MiniTensors)| {
            let fx = fixture(tag, heads, latent, mutate, retype);
            let split = open(&fx);
            let checked = check_kinds(&spec(), &split, &src_split);
            drop(split);
            let _ = std::fs::remove_file(&fx);
            checked
        };
        check("muse-fx-ok", 16, 512, &|_| {}, &|_| {})
            .expect("the mini fixture of the map's four layers reads");

        let flip = |kv: &mut MiniKv| {
            set(
                kv,
                "attention.sliding_window_pattern",
                V::Bools(vec![true; 4]),
            );
        };
        let err = check("muse-fx-flip", 16, 512, &flip, &|_| {}).expect_err("every layer sliding");
        assert!(err.to_string().contains("layer 1 (source 1)"), "{err}");

        let err = check("muse-fx-heads", 128, 512, &|_| {}, &|_| {})
            .expect_err("the source's head count");
        assert!(
            err.to_string()
                .contains("attention.head_count: 128, the mini writes 16"),
            "{err}"
        );

        let err =
            check("muse-fx-latent", 16, 4096, &|_| {}, &|_| {}).expect_err("the source's latent");
        assert!(
            err.to_string()
                .contains("moe_latent_size: 4096, the mini writes 512"),
            "{err}"
        );

        let softcap = |kv: &mut MiniKv| {
            set(kv, "final_logit_softcapping", V::F32(30.0));
        };
        let err =
            check("muse-fx-softcap", 16, 512, &softcap, &|_| {}).expect_err("another softcap");
        assert!(
            err.to_string()
                .contains("final_logit_softcapping: 30, the source's 20"),
            "{err}"
        );

        let retype = |t: &mut MiniTensors| {
            t.iter_mut()
                .find(|(n, _, _)| n == "blk.2.attn_q.weight")
                .expect("attn_q is in the fixture")
                .2 = Ty::Q8_0;
        };
        let err =
            check("muse-fx-type", 16, 512, &|_| {}, &retype).expect_err("a Q8_0 q projection");
        assert!(
            err.to_string()
                .contains(&format!("attn_q.weight is {:?}", GgmlType::Q8_0)),
            "{err}"
        );

        let embd = |t: &mut MiniTensors| {
            t.iter_mut()
                .find(|(n, _, _)| n == "token_embd.weight")
                .expect("token_embd is in the fixture")
                .2 = Ty::F32;
        };
        let err = check("muse-fx-embd", 16, 512, &|_| {}, &embd).expect_err("an F32 embedding");
        assert!(
            err.to_string()
                .contains(&format!("token_embd.weight: is {:?}", GgmlType::F32)),
            "{err}"
        );
        drop(src_split);
        let _ = std::fs::remove_file(&src);
    }
}
