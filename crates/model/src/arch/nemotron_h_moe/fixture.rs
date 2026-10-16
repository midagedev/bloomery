//! The Nemotron 3.5 Bolt 74B-A9B (`nemotron_h_moe`) gate fixture's spec
//! ([`spec`]): the facts of a `nemotron_h_moe` file the common generator
//! (`crate::fixture`) cannot know. The model is a hybrid of Mamba-2, attention
//! and latent MoE layers; the MoE layers differ from layer to layer (each its
//! own expert count and width), there is no next-token layer, and the source is
//! one unsplit file.
//!
//! The fixture is four blocks and the source's globals. Every layer is a
//! source layer, renamed `blk.L.` → `blk.f.`; no expert is dropped, so a MoE
//! layer keeps its own count and width:
//!
//! | fixture layer | source layer | mixer | feed-forward | why |
//! |---|---|---|---|---|
//! | f0 | L0 | Mamba-2 | none | the first Mamba-2 layer: `ssm_in`/`ssm_out` NVFP4 with their F32 `.scale`, the F32 `ssm_*` vectors, conv and grouped norm |
//! | f1 | L3 | none | latent MoE, 240 experts × 1536 | the least expert count and the least width of the file's twelve (count, width) pairs |
//! | f2 | L7 | attention, 2 KV heads | none | the first attention layer: `attn_{q,k,v,output}` Q8_0, group 16 |
//! | f3 | L48 | none | latent MoE, 496 experts × 2560 | the greatest count (the file's `expert_count`) and the greatest width |
//!
//! What it keeps: every (tensor stem, type) pair of the source's layers (one
//! type per stem: NVFP4 `ssm_in`/`ssm_out`, IQ2_XXS routed and shared experts,
//! Q8_0 attention and latent pair, F32 router, bias and vectors), one layer of
//! each kind the engine's launch table is built from, and two MoE layers whose
//! (expert count, width) pairs differ, so the per-layer shapes are not one
//! shape. What it drops: 39 Mamba-2 layers, 38 MoE layers and 7 attention
//! layers. [`check_map`] refuses a map that loses a kind, a pair, or the
//! difference between two MoE layers, by name, at the plan and at `verify`.
//!
//! The reader ([`Hparams::read`]) takes the layer kinds from the two per-layer
//! arrays and the marker tensors, and the expert counts from the routers'
//! rows; the fixture keeps the four per-layer arrays at the map's layers
//! (`attention.head_count_kv`, `feed_forward_length`,
//! `expert_feed_forward_length` and `expert_used_count`), so a kind and an
//! expert shape move with their layer. `expert_count` is the largest layer's
//! router rows, which the reader checks, so it is a table key ([`Tables`]): the
//! largest of the map's MoE layers.
//!
//! The tensor data is 2,158,358,920 bytes in 37 tensors [derived from the
//! header's tensor sizes]: 1.14 GB of it the two Q8_0 globals (`token_embd`,
//! `output`), 0.89 GB IQ2_XXS and 0.06 GB NVFP4.
//!
//! The model is one file and the fixture is one file: the source carries no
//! `split.*` keys, and the shard cap is the generator's default, far above the
//! fixture's size, so the plan writes none (`plan.rs` `shard`). No MTP file
//! exists to fixture.
//!
//! What the Nano's fixture (`nemotron_h::fixture`) shares is in
//! `crate::fixture::hybrid`: the trunk's key rules and constants, the map's
//! kind and pair coverage, and the trunk comparison. This file holds what the
//! latent MoE forces: the expert arrays and table key, the router and bias
//! constants, the NVFP4 per-tensor scales, and the map's two-shape rule.

use std::collections::BTreeSet;

use gguf::{Split, Value};

use super::hparams::{Experts, Hparams, Kind};
use super::names::{EXP_PROBS_B, SSM_IN_SCALE, SSM_OUT_SCALE};
use crate::fixture::hybrid::{self, Scan, Trunk};
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec, KeyRule,
    NVFP4_TENSOR_SCALE, Tables, Window, int_like, meta,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [0, 3, 7, 48];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at
/// every K the fixture holds (4,096 for the Q8_0 projections, globals and
/// NVFP4 `ssm_in`, 8,192 for NVFP4 `ssm_out`, 1,024 for the latent up and the
/// routed up stack, 1,536 and 2,560 for the routed down stack, 4,096 and 5,376
/// for the shared expert). For NVFP4 the window holds the product of the
/// tensor's scale and the sub-block scale.
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The card byte budget the fixture records. A budget above the fixture's
/// tensor data (about 2.16 GB) keeps the whole model resident. The engine
/// attach sets the value its placement reads.
pub const CARD_BUDGET: u64 = 4 << 30;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "nemotron-bolt-fixture";

/// The real file `verify` opens when none is named: the profile of the box.
pub const DEFAULT_MODEL: &str =
    "/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf";

const ARCH: &str = "nemotron_h_moe";

/// The scan's constants: the real file's medians over its 40 Mamba-2 layers'
/// 128 heads each, −0.389, 0.220 and −3.0 [measured]. The heads' `ssm_a` run
/// from −6,718 to −0.0011 (the 1 % and 99 % quantiles are −173.5 and −0.0092),
/// and a constant is the median.
const SCAN: Scan = Scan {
    a: -0.39,
    d: 0.22,
    dt_bias: -3.0,
};

/// The Nemotron 3.5 Bolt fixture: [`LAYER_MAP`], the window
/// [[`D_MIN`], [`D_MAX`]], the box's file as the default source, no draft, no
/// ff override (the experts keep the layers' own widths) and a fixed card
/// budget.
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
        family: &Bolt,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does
/// not cover. The hybrid trunk's ([`hybrid::trunk_key_rule`]), and the latent
/// MoE's: the two per-layer expert arrays follow the map, `expert_count` is a
/// table key (the largest router of the map), and the router's and shared
/// expert's constants are copied.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 8] = [
        "expert_group_count",
        "expert_group_used_count",
        "expert_gating_func",
        "expert_shared_feed_forward_length",
        "expert_shared_count",
        "expert_weights_norm",
        "expert_weights_scale",
        "moe_latent_size",
    ];
    match suffix {
        "expert_feed_forward_length" | "expert_used_count" => Some(KeyRule::PerLayer),
        "expert_count" => Some(KeyRule::Table),
        s if COPY.contains(&s) => Some(KeyRule::Copy),
        s => hybrid::trunk_key_rule(s),
    }
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the
/// trunk's ([`hybrid::trunk_const`]) at [`SCAN`]; the two NVFP4 per-tensor
/// scales, [`NVFP4_TENSOR_SCALE`] (the scale the generator writes the NVFP4
/// blocks at). The router's selection bias is a spread ([`spread_value`]).
pub fn const_value(leaf: &str) -> Option<f32> {
    match leaf {
        SSM_IN_SCALE | SSM_OUT_SCALE => Some(NVFP4_TENSOR_SCALE),
        _ => hybrid::trunk_const(leaf, &SCAN),
    }
}

/// The half-width of the router's selection bias (`exp_probs_b`), drawn per
/// expert. A constant bias moves no pick, so a body that skipped the add would
/// pass; the real file's biases (4.44 to 51.25 [measured]) would fix the picks
/// whatever the scores. ±0.05 (σ 0.029) sits under the sigmoid scores' range
/// and against about 0.01 between the ninth and tenth score at the top of 240
/// to 496 [derived: unit-variance logits, the order statistics' spacing times
/// the sigmoid's slope there], so the add moves the top-9 on some positions and
/// fixes none.
pub const BIAS_HALF_WIDTH: f32 = 0.05;

/// The spread of an F32 tensor the family names: the router's selection bias
/// ([`BIAS_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    (leaf == EXP_PROBS_B).then_some(BIAS_HALF_WIDTH)
}

/// `layers` keeps the source's facts: one layer of each kind (Mamba-2,
/// attention, MoE), every (stem, type) pair any source layer holds
/// ([`hybrid::check_map`]), and two MoE layers of different (expert count,
/// width): a map of one shape would not tell the engine's per-layer shapes
/// from a model-wide one. The map's largest router, the fixture's `expert_count`.
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<usize, FixtureError> {
    hybrid::check_map(
        &src.kinds,
        &[Kind::Mamba, Kind::Attention, Kind::Moe],
        source,
        layers,
    )?;
    let shapes: BTreeSet<(usize, usize)> = moe_layers(src, layers)
        .map(|e| (e.n_expert, e.ff))
        .collect();
    if shapes.len() < 2 {
        return Err(hybrid::map_error(format!(
            "{layers:?} holds MoE layers of {} (expert count, width) shape(s) {shapes:?}: two different are kept",
            shapes.len()
        )));
    }
    Ok(moe_layers(src, layers)
        .map(|e| e.n_expert)
        .max()
        .expect("two shapes hold a MoE layer"))
}

/// The experts of the MoE layers in `layers`.
fn moe_layers<'a>(src: &'a Hparams, layers: &'a [usize]) -> impl Iterator<Item = Experts> + 'a {
    layers
        .iter()
        .filter_map(|&l| src.experts.get(l).copied().flatten())
}

/// Latent MoE's rules.
struct Bolt;

impl Family for Bolt {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// Only `expert_count` moves with the map, and the map is held to
    /// [`check_map`].
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let src = Hparams::read(source)?;
        let n_expert = check_map(&src, source, &spec.layers)?;
        Ok(Box::new(MapTables { n_expert }))
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

/// What the map decides: `expert_count`, the largest router of its MoE layers.
struct MapTables {
    n_expert: usize,
}

impl Tables for MapTables {
    fn key(&self, key: &str, suffix: &str, v: &Value) -> Result<Value, FixtureError> {
        match suffix {
            "expert_count" => int_like(key, v, self.n_expert as u64),
            _ => Err(meta(
                key,
                "is a table key the nemotron_h_moe map does not compute",
            )),
        }
    }

    fn lines(&self) -> Vec<String> {
        vec![format!(
            "nemotron_h_moe map: expert_count {}, the largest router of its MoE layers",
            self.n_expert
        )]
    }
}

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds, KV head
/// counts and experts (count, top-k, width) are the
/// source layers', at the source's globals, Mamba-2 widths and latent MoE
/// constants, with `expert_count` the map's largest router, under a map
/// [`check_map`] accepts. Tensor names, dims and types are the plan's, which
/// `verify` holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    let largest = check_map(&src, source, &spec.layers)?;
    hybrid::check_trunk(&spec.layers, &trunk(&fx), &trunk(&src))?;
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    for (f, &l) in spec.layers.iter().enumerate() {
        if fx.experts[f] != src.experts[l] {
            return Err(bad(
                format!("layer {f} (source {l})"),
                format!(
                    "experts {:?}, the source's {:?}",
                    fx.experts[f], src.experts[l]
                ),
            ));
        }
    }
    if largest != fx.n_expert {
        return Err(bad(
            "expert_count".into(),
            format!("{}, the map's largest router is {largest}", fx.n_expert),
        ));
    }
    for (key, a, b) in [
        ("moe_latent_size", fx.latent, src.latent),
        (
            "expert_shared_feed_forward_length",
            fx.shared_ff,
            src.shared_ff,
        ),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, want {b}")));
        }
    }
    if fx.weights_norm != src.weights_norm {
        return Err(bad(
            "expert_weights_norm".into(),
            format!("{}, the source's {}", fx.weights_norm, src.weights_norm),
        ));
    }
    if fx.weights_scale.to_bits() != src.weights_scale.to_bits() {
        return Err(bad(
            "expert_weights_scale".into(),
            format!("{}, the source's {}", fx.weights_scale, src.weights_scale),
        ));
    }
    Ok(fx)
}

/// The facts of `hp` a fixture keeps in common with the other hybrid family.
fn trunk(hp: &Hparams) -> Trunk<'_, Kind> {
    Trunk {
        n_layer: hp.n_layer,
        n_embd: hp.n_embd,
        n_ctx: hp.n_ctx,
        n_vocab: hp.n_vocab,
        n_head: hp.n_head,
        key_len: hp.key_len,
        value_len: hp.value_len,
        rms_eps: hp.rms_eps,
        ssm: &hp.ssm,
        kv: &hp.kv,
        ff: None,
        kinds: &hp.kinds,
    }
}
