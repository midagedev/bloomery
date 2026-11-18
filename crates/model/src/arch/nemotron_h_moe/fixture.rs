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
//!
//! The second fixture of the architecture is Nemotron 3.5 Lightning's
//! ([`light_spec`], the profile `nemotron-light`): the same rules over a
//! 30B-A3B file whose experts act on the hidden row itself (no latent), 128
//! of them 1,856 wide at the top 6, in Q5_0, Q5_1 and Q8_0 blocks, with Q8_0
//! mixers and a next-token block the fixture leaves out. It is four blocks of
//! source layers 0, 1, 3 and 5:
//!
//! | fixture layer | source layer | mixer | feed-forward | why |
//! |---|---|---|---|---|
//! | f0 | L0 | Mamba-2 | none | the first Mamba-2 layer: Q8_0 `ssm_in`/`ssm_out` |
//! | f1 | L1 | none | MoE, up Q5_1, down Q8_0 | the one layer of the file with a Q5_1 up (the other 22 hold Q5_0), and the Q8_0 down four of its layers hold |
//! | f2 | L3 | none | MoE, up Q5_0, down Q5_1 | the pair the other 18 layers hold |
//! | f3 | L5 | attention, 2 KV heads | none | the first attention layer: Q8_0 `attn_{q,k,v,output}` |
//!
//! Both MoE layers are one (expert count, width) shape, as the file's 23 are,
//! so the map's two-shape rule is one shape here (`Moe::min_shapes`); every
//! (stem, type) pair of the trunk's layers is kept (`hybrid::check_map_trunk`,
//! which leaves the next-token block's own pairs, its BF16 router among them,
//! out of the file's). The scan's constants are the Bolt's: the Lightning's
//! medians are not measured, and the fixture needs the type, not the quality.

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
        family: &BOLT,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// Fixture layer `f` of the Lightning fixture holds source layer `LIGHT_LAYER_MAP[f]` (the table
/// above).
pub const LIGHT_LAYER_MAP: [usize; 4] = [0, 1, 3, 5];

/// The Lightning fixture is `<LIGHT_STEM>-00001-of-00001.gguf`.
pub const LIGHT_STEM: &str = "nemotron-lightning-fixture";

/// The real file `verify` opens for the Lightning fixture when none is named: the Unsloth
/// UD-Q4_K_XL on the box.
pub const LIGHT_DEFAULT_MODEL: &str =
    "/models/Nemotron-3.5-Lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf";

/// The Nemotron 3.5 Lightning fixture: [`LIGHT_LAYER_MAP`] under the Bolt's window, card budget
/// and rules, the one-shape map rule, and the Lightning's file as the default source.
pub fn light_spec() -> FixtureSpec {
    FixtureSpec {
        stem: LIGHT_STEM,
        layers: LIGHT_LAYER_MAP.to_vec(),
        default_source: light_default_source,
        family: &LIGHTNING,
        ..spec()
    }
}

fn light_default_source() -> String {
    LIGHT_DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does
/// not cover. The hybrid trunk's ([`hybrid::trunk_key_rule`]), and the latent
/// MoE's: the two per-layer expert arrays follow the map, `expert_count` is a
/// table key (the largest router of the map), and the router's and shared
/// expert's constants are copied.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 9] = [
        "rope.freq_base",
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
        // The fixture maps the source's trunk layers and carries no
        // next-token block.
        "nextn_predict_layers" => Some(KeyRule::Set(0)),
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
    check_map_shapes(src, source, layers, BOLT.min_shapes)
}

/// [`check_map`] with the number of different (expert count, width) shapes the map must keep: the
/// source's own, at most (a source whose MoE layers are all of one shape keeps one).
fn check_map_shapes(
    src: &Hparams,
    source: &Split,
    layers: &[usize],
    min_shapes: usize,
) -> Result<usize, FixtureError> {
    hybrid::check_map_trunk(
        &src.kinds,
        &[Kind::Mamba, Kind::Attention, Kind::Moe],
        source,
        layers,
        src.n_trunk,
    )?;
    let shapes: BTreeSet<(usize, usize)> = moe_layers(src, layers)
        .map(|e| (e.n_expert, e.ff))
        .collect();
    let trunk: Vec<usize> = (0..src.n_trunk).collect();
    let source_shapes: BTreeSet<(usize, usize)> = moe_layers(src, &trunk)
        .map(|e| (e.n_expert, e.ff))
        .collect();
    let want = min_shapes.min(source_shapes.len());
    if shapes.len() < want {
        return Err(hybrid::map_error(format!(
            "{layers:?} holds MoE layers of {} (expert count, width) shape(s) {shapes:?}: {want} different are kept",
            shapes.len()
        )));
    }
    Ok(moe_layers(src, layers)
        .map(|e| e.n_expert)
        .max()
        .expect("a kept shape holds a MoE layer"))
}

/// The experts of the MoE layers in `layers`.
fn moe_layers<'a>(src: &'a Hparams, layers: &'a [usize]) -> impl Iterator<Item = Experts> + 'a {
    layers
        .iter()
        .filter_map(|&l| src.experts.get(l).copied().flatten())
}

/// The MoE hybrid's rules: the Bolt's and the Lightning's differ in the shapes a map must keep.
struct Moe {
    /// The different (expert count, width) shapes a map keeps, where its source holds that many.
    min_shapes: usize,
}

/// The Bolt's: its twelve shapes are not one shape, and a map of one would not tell the per-layer
/// shapes from a model-wide one.
static BOLT: Moe = Moe { min_shapes: 2 };

/// The Lightning's: all its MoE layers are one shape.
static LIGHTNING: Moe = Moe { min_shapes: 1 };

impl Family for Moe {
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
    /// [`check_map`]. An NVFP4 tensor the map holds is refused without its
    /// companion `.scale` at [`NVFP4_TENSOR_SCALE`]: the fill divides its
    /// target σ by the tensor scale, so a missing scale would write the
    /// projections at the wrong RMS.
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let src = Hparams::read(source)?;
        let n_expert = check_map_shapes(&src, source, &spec.layers, self.min_shapes)?;
        for (_, t) in source.iter_tensors() {
            if t.ty != gguf::GgmlType::NVFP4 {
                continue;
            }
            if let Some((l, _)) = crate::fixture::split_layer(&t.name)
                && !spec.layers.contains(&l)
            {
                continue;
            }
            let scale = t
                .name
                .strip_suffix(".weight")
                .map(|name| format!("{name}.scale"));
            let leaf = scale
                .as_deref()
                .map(|name| crate::fixture::split_layer(name).map_or(name, |(_, leaf)| leaf));
            if scale
                .as_deref()
                .and_then(|name| source.find(name))
                .is_none()
                || leaf.and_then(const_value) != Some(NVFP4_TENSOR_SCALE)
            {
                return Err(FixtureError::Tensor {
                    name: t.name.clone(),
                    detail: "NVFP4 fill requires a companion .scale tensor at NVFP4_TENSOR_SCALE"
                        .into(),
                });
            }
        }
        Ok(Box::new(MapTables { n_expert }))
    }

    fn check_kinds(
        &self,
        spec: &FixtureSpec,
        fixture: &Split,
        source: &Split,
    ) -> Result<(), FixtureError> {
        check_kinds_shapes(spec, fixture, source, self.min_shapes).map(|_| ())
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
    check_kinds_shapes(spec, fixture, source, BOLT.min_shapes)
}

/// [`check_kinds`] under a map that keeps `min_shapes` expert shapes ([`check_map_shapes`]).
fn check_kinds_shapes(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
    min_shapes: usize,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    let largest = check_map_shapes(&src, source, &spec.layers, min_shapes)?;
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
            Some(fx.shared_ff),
            Some(src.shared_ff),
        ),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a:?}, want {b:?}")));
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
        n_layer: hp.n_trunk,
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
