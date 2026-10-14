//! The Nemotron 3 Nano 4B (`nemotron_h`) gate fixture's spec ([`spec`]): the
//! facts of a `nemotron_h` file the common generator (`crate::fixture`) cannot
//! know. The model is dense: no routed experts, no next-token layer, one
//! unsplit file.
//!
//! The fixture is five blocks and the source's globals. Every layer is a
//! source layer, renamed `blk.L.` → `blk.f.`:
//!
//! | fixture layer | source layer | mixer | feed-forward | why |
//! |---|---|---|---|---|
//! | f0 | L0 | Mamba-2 | none | the first Mamba-2 layer: `ssm_in` Q5_0, `ssm_out` Q4_K, the F32 `ssm_*` vectors, conv and grouped norm |
//! | f1 | L1 | none | squared-ReLU MLP, `ffn_down` Q6_K | the first MLP layer |
//! | f2 | L12 | attention, 8 KV heads | none | the first attention layer: `attn_v` Q5_0 |
//! | f3 | L13 | none | MLP, `ffn_down` Q4_K | the first MLP layer with the other `ffn_down` type |
//! | f4 | L24 | attention, 8 KV heads | none | `attn_v` Q8_0, the other `attn_v` type (L32 is the same) |
//!
//! What it keeps: every (tensor stem, type) pair of the source's layers
//! (`ssm_in`, `ffn_up`, `attn_q`, `attn_k` and `attn_v` Q5_0 or Q8_0, `ssm_out`
//! and `attn_output` Q4_K, `ffn_down` Q6_K or Q4_K), and one layer of each kind
//! the engine's launch table is built from. What it drops: 20 Mamba-2 layers, 15
//! MLP layers and 2 attention layers. [`check_map`] refuses a map that loses a
//! kind or a pair, by name, at the plan and at `verify`.
//!
//! The reader ([`Hparams::read`]) takes the layer kinds from the two per-layer
//! arrays and the marker tensors; the fixture keeps both arrays at the map's
//! layers (`attention.head_count_kv` and `feed_forward_length` are per-layer
//! keys), so a kind moves with its layer.
//!
//! The model is one file and the fixture is one file: the source carries no
//! `split.*` keys, and the shard cap is the generator's default, far above the
//! fixture's size, so the plan writes none (`plan.rs` `shard`). No MTP file
//! exists to fixture.

use std::collections::BTreeMap;

use gguf::Split;

use super::hparams::{Hparams, Kind};
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec, KeyRule, NoTables, Tables,
    Window, split_layer,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 5] = [0, 1, 12, 13, 24];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at
/// every K the fixture holds (3,136 for Q5_0 and Q8_0, 5,120 and 7,680 for
/// Q4_K, 12,544 for Q4_K and Q6_K; Q6_K at 12,544 needs a narrow band to keep
/// `d` above 2^-14).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The card byte budget the fixture records. The model has no routed experts,
/// so nothing in the file plans against it, and a budget above the fixture's
/// tensor data (about 0.93 GB) keeps the whole model resident. The engine
/// attach sets the value its placement reads.
pub const CARD_BUDGET: u64 = 1 << 30;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "nemotron-nano-fixture";

/// The real file `verify` opens when none is named: the profile of the box.
pub const DEFAULT_MODEL: &str =
    "/models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf";

const ARCH: &str = "nemotron_h";

/// The `ssm_a` constant: the real file's median over its 21 Mamba-2 layers'
/// 96 heads each is −0.305 [measured]. It is stored as `−exp(A_log)`, so every
/// head's is negative (the largest is −8.4e-6) and `dA = exp(dt·A)` stays
/// below 1 for any `dt > 0`; the heads span four decades (the 1 % and 99 %
/// quantiles are −2,714 and −0.0048), and a constant is the median.
const SSM_A: f32 = -0.30;
/// The `ssm_d` constant, the real file's median, 0.664 [measured].
const SSM_D: f32 = 0.66;
/// The `ssm_dt.bias` constant, the real file's median, −2.86 [measured]: with
/// `dt = softplus(dt_raw − 2.86)` the step is about 0.06 to 0.2 and the decay
/// `exp(dt·A)` sits near 1, as the real file's does.
const SSM_DT_BIAS: f32 = -2.86;

/// The Nemotron 3 Nano 4B fixture: [`LAYER_MAP`], the window
/// [[`D_MIN`], [`D_MAX`]], the box's file as the default source, no draft, no
/// ff override (the model is dense) and a fixed card budget (a planned one
/// needs routed experts).
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
        family: &Nemo,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does
/// not cover. The copy list is every key the file carries that is not a
/// function of the map; `nextn_predict_layers` may be present and must be 0
/// (the reader runs no next-token layer).
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 15] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "attention.layer_norm_epsilon",
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
        "rope.scaling.finetuned",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.inner_size",
        "ssm.time_step_rank",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count_kv" | "feed_forward_length" => KeyRule::PerLayer,
        "nextn_predict_layers" => KeyRule::Zero("the reader runs no next-token layer"),
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the
/// norms' gains 1, including the grouped `ssm_norm` (a `[d_inner / groups,
/// groups]` matrix, still a gain), the conv's bias 0 (the real file's median is
/// 8e-5 [measured]), and the scan's `ssm_a`, `ssm_d` and `ssm_dt.bias`
/// ([`SSM_A`], [`SSM_D`], [`SSM_DT_BIAS`]). The scan's vectors are stored
/// `[1, heads]` (`ssm_a`, `ssm_d`) and `[heads]`; every head shares each
/// constant.
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 7] = [
        ("attn_norm.weight", 1.0),
        ("output_norm.weight", 1.0),
        ("ssm_norm.weight", 1.0),
        ("ssm_conv1d.bias", 0.0),
        ("ssm_a", SSM_A),
        ("ssm_d", SSM_D),
        ("ssm_dt.bias", SSM_DT_BIAS),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// Every (tensor stem, type name) of `split`'s layer tensors with the first
/// layer that holds the pair, for the layers `keep` accepts.
fn pairs(split: &Split, keep: impl Fn(usize) -> bool) -> BTreeMap<(String, String), usize> {
    let mut out = BTreeMap::new();
    for (_, t) in split.iter_tensors() {
        if let Some((l, leaf)) = split_layer(&t.name)
            && keep(l)
        {
            out.entry((leaf.to_string(), t.ty.to_string())).or_insert(l);
        }
    }
    out
}

/// `layers` keeps the source's facts: one layer of each kind (Mamba-2,
/// attention, MLP), and every (stem, type) pair any source layer holds. A
/// quant type that appears in the source on one layer only (`attn_v` Q8_0 on
/// two of four, `ffn_down` Q4_K on eight of seventeen) is a kernel dispatch
/// the gate would not reach if the map dropped it, and a kind the map drops is
/// a launch the fixture never makes.
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    let bad = |what: &str, detail: String| FixtureError::Mismatch {
        what: what.to_string(),
        detail,
    };
    let kinds: Vec<Kind> = layers
        .iter()
        .filter_map(|&l| src.kinds.get(l).copied())
        .collect();
    for kind in [Kind::Mamba, Kind::Attention, Kind::Mlp] {
        if !kinds.contains(&kind) {
            return Err(bad(
                "layer map",
                format!(
                    "{layers:?} holds no {kind:?} layer (kinds {kinds:?}): one of each is kept"
                ),
            ));
        }
    }
    let held = pairs(source, |l| layers.contains(&l));
    if let Some(((leaf, ty), l)) = pairs(source, |_| true)
        .into_iter()
        .find(|(pair, _)| !held.contains_key(pair))
    {
        return Err(bad(
            "layer map",
            format!("{layers:?} holds no {leaf} of type {ty}, which source layer {l} does"),
        ));
    }
    Ok(())
}

/// nemotron_h's rules.
struct Nemo;

impl Family for Nemo {
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

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds, KV head
/// counts and feed-forward lengths are the source layers', at the source's
/// globals and Mamba-2 widths, under a map [`check_map`] accepts. Tensor names,
/// dims and types are the plan's, which `verify` holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    if fx.n_layer != spec.layers.len() {
        return Err(bad(
            "layer count".into(),
            format!("{}, not {}", fx.n_layer, spec.layers.len()),
        ));
    }
    check_map(&src, source, &spec.layers)?;
    for (f, &l) in spec.layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        for (key, a, b) in [
            ("kv heads", fx.kv[f], src.kv[l]),
            ("feed-forward length", fx.ff[f], src.ff[l]),
        ] {
            if a != b {
                return Err(bad(what, format!("{key} {a}, the source's {b}")));
            }
        }
        if fx.kinds[f] != src.kinds[l] {
            return Err(bad(
                what,
                format!("kind {:?}, the source's {:?}", fx.kinds[f], src.kinds[l]),
            ));
        }
    }
    for (key, a, b) in [
        ("n_embd", fx.n_embd, src.n_embd),
        ("n_ctx", fx.n_ctx, src.n_ctx),
        ("n_vocab", fx.n_vocab, src.n_vocab),
        ("n_head", fx.n_head, src.n_head),
        ("key_len", fx.key_len, src.key_len),
        ("value_len", fx.value_len, src.value_len),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, want {b}")));
        }
    }
    if fx.ssm != src.ssm {
        return Err(bad(
            "ssm".into(),
            format!("{:?}, the source's {:?}", fx.ssm, src.ssm),
        ));
    }
    if fx.rms_eps.to_bits() != src.rms_eps.to_bits() {
        return Err(bad(
            "attention.layer_norm_rms_epsilon".into(),
            format!("{}, the source's {}", fx.rms_eps, src.rms_eps),
        ));
    }
    Ok(fx)
}
