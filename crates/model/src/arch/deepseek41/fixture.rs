//! The V4.1 gate fixture's spec ([`spec`]): the facts of a V4.1 file the
//! common generator (`crate::fixture`) cannot know.
//!
//! The fixture is nine layers, one per layer kind the step distinguishes
//! ([`LAYER_MAP`], whose compress ratios must read [`FIXTURE_RATIOS`]), and
//! the source's globals. Every block's `d` (and `dmin`) lies in
//! [[`D_MIN`], [`D_MAX`]]. Every key of the coverage table ([`key_rule`]) keeps
//! its source's value but the layer count, the per-layer arrays, the engram
//! layer ids, the engram primes and offsets, and the routed experts' ff: an
//! engram table's rows follow the fixture's own hash primes, the smallest above
//! [`ENGRAM_PRIME_FLOOR`], so a table is a few hundred thousand rows instead
//! of several hundred million, and the ff is [`FIXTURE_FF`] instead of the
//! source's 2,304, on every routed stack and on the shared expert, which the
//! chain sizes at the same ff ([`Family::ff_axis`]). A 1-D F32 tensor holds
//! its constant ([`const_value`]): gains 1, sinks, biases and hyper-connection
//! bases 0, hyper-connection scales 1; the one exception is `exp_probs_b_vl`,
//! a spread ([`VL_BIAS_HALF_WIDTH`]): a bias that is the same on every expert
//! moves no router pick, and the media gate's coverage clause needs the
//! media positions' picks to differ from the text positions'.
//!
//! The DSpark draft fixture ([`DRAFT_FILE`]) is the real draft's tensors at
//! their shapes with random weights by the same rules, its ff narrowed with
//! the target's (the real draft has the target's ff), `target_layers` moved
//! to the fixture's last layers, and the generator's five keys.
//!
//! The host tier reads an r8 sidecar of the routed gate and up stacks; the
//! fixture gets its own ([`r8_stacks`]), written beside it by `generate`
//! and checked by `verify`.
//!
//! The card budget is planned ([`Budget`]): under it the written file's own
//! plan, on the gate machine at [`workstation::CTX_MAX`] positions, holds half
//! of every layer's experts on the card, 192 of 384.

use gguf::{GgmlType, Split, Value};

use super::hparams::{DenseStream, Experts, Hparams, LayerKind, Stream};
use super::names;
use super::place::PlanInputs;
use crate::arch::dspark::{self, DraftHparams};
use crate::fixture::{
    Budget, CardBudget, CardExperts, DraftRules, DraftSpec, Family, FixtureError, FixtureSpec,
    KeyRule, Kvs, SidecarSpec, Tables, Window, int_like, items, meta, unsigned,
};
use crate::placement::PlanLevers;
use crate::placement::workstation;

/// Fixture layer `f` holds source layer `LAYER_MAP[f]`: one layer per kind
/// the step's launch table tells apart (window, window+engram, the r2 source
/// and reader, the r2 source with engram, the r1 source that ends the
/// compressed layers, the r1 readers with and without an indexer), and a
/// last reader whose top-k source is not its kv source.
pub const LAYER_MAP: [usize; 9] = [0, 1, 2, 3, 14, 20, 21, 24, 25];

/// The compress ratios the map must read from the source: the check that
/// the map still picks the kinds it was chosen for.
pub const FIXTURE_RATIOS: [u64; 9] = [0, 0, 2, 2, 2, 1, 1, 1, 1];

/// The d window, [2^-13, 2^-9]: normal f16 values every V4.1 type's scale
/// fits at every K the fixture holds. The upper bound is 2^-9, not the 2^-10 it
/// was while the fixture kept the source's ff: at ff 512 a down stack's K is
/// 512, σ = 0.044, and a Q4_K or Q5_K block with `dmin = 7.5·d` (7.75·d) and
/// the widest scale band needs `dmin` about 2^-9.7 to 2^-8.9, past 2^-10.
pub const D_MIN: f32 = 1.0 / 8192.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 512.0;

/// The routed experts' ff the fixture writes in place of the source's 2,304,
/// on the routed stacks, the shared expert and the draft: a multiple of 256
/// (`gpu-deepseek41`'s `chain/ffn.rs` refuses any other).
pub const FIXTURE_FF: u64 = 512;

/// The positions the card budget's plan runs at: the V4.1 gates' context
/// (`workstation::CTX_MAX`), the one every V4.1 body gate plans at.
pub const BUDGET_CTX: u64 = workstation::CTX_MAX;

/// The most tensor data a target shard holds. The fixture is about 15 GB, and
/// the real file is a nine-shard split set: a cap of 4 GiB cuts it into
/// shards with a layer across a boundary, so the engine reads a split set
/// here too.
pub const SHARD_BYTES: u64 = 4 << 30;

/// The half-width of `exp_probs_b_vl`'s spread. The real file's text and media
/// biases differ by a constant, which moves no pick, and a spread of standard
/// deviation 0.04 to 0.08 [measured, the header's blocks 2 and 14]; the
/// fixture's ±0.1 has 0.058, against about 0.03 between the sixth and seventh
/// router score at the top of 384 [derived: sqrt-softplus of unit-variance
/// logits], so most positions pick differently under it.
pub const VL_BIAS_HALF_WIDTH: f32 = 0.1;

/// The least spread (max − min over the experts) of `exp_probs_b_vl −
/// exp_probs_b` a written layer holds: half the spread's expected 2·h. A
/// layer under it is one whose media picks cannot differ from its text picks.
pub const BIAS_SPREAD_MIN: f32 = VL_BIAS_HALF_WIDTH;

/// The target shards are `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "v41-fixture";
/// The draft fixture's path inside the fixture directory: a directory of its
/// own, so a reader that takes every `*.gguf` beside the target's shards (the
/// engram tables' `shards_in`) never meets it.
pub const DRAFT_FILE: &str = "draft/v41-fixture-draft.gguf";

/// Engram hash primes are the smallest primes above this, as many as the
/// sites' buckets, in site order.
pub const ENGRAM_PRIME_FLOOR: u64 = 1 << 14;

const TARGET_ARCH: &str = "deepseek41";

/// The V4.1 fixture: [`LAYER_MAP`], [`FIXTURE_RATIOS`], the window
/// [[`D_MIN`], [`D_MAX`]], [`FIXTURE_FF`], the V4.1 file as the default
/// source, the DSpark draft, the r8 sidecar and the planned card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: TARGET_ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: FIXTURE_RATIOS.to_vec(),
        card_budget: CardBudget::Planned(Budget {
            ctx: BUDGET_CTX,
            card_experts,
        }),
        window: Window::new(D_MIN, D_MAX).expect("[2^-13, 2^-9] is inside the normal f16 values"),
        ff: Some(FIXTURE_FF),
        shard_bytes: SHARD_BYTES,
        default_source: gguf::v41::model,
        draft: Some(DraftSpec {
            arch: crate::arch::DFLASH,
            file: DRAFT_FILE,
            default_source: default_draft_source,
            rules: &DSpark,
        }),
        sidecar: Some(SidecarSpec { stacks: r8_stacks }),
        family: &V41,
    }
}

/// The written file's per-layer card expert counts, planned through the
/// placement planner on the gate card ([`workstation::plan_gate`], the 3090)
/// at [`BUDGET_CTX`] positions under `budget` bytes — the plan every V4.1
/// body gate makes of a file it opens. `describe`, not `read`: the gates'
/// plan of a file needs no feature check to count experts.
fn card_experts(split: &Split, budget: Option<u64>) -> Result<CardExperts, FixtureError> {
    let inputs = PlanInputs::describe(split)?;
    let machine = workstation::plan_gate(inputs.model.layers);
    let levers = PlanLevers {
        card_budget_bytes: budget,
    };
    let plan = inputs
        .plan(&machine, BUDGET_CTX, &levers)
        .map_err(|e| FixtureError::Budget(e.to_string()))?;
    Ok(CardExperts {
        per_layer: plan.n_l,
        experts: inputs.model.experts,
    })
}

/// The stacks the r8 sidecar holds: the gate and up stacks of every layer
/// `hparams` reads as routed, in layer order — what `r8conv convert` takes of
/// the real file (`bin/r8conv.rs`, `routed_stacks`).
fn r8_stacks(fixture: &Split) -> Result<Vec<String>, FixtureError> {
    let hp = Hparams::read(fixture)?;
    Ok(hp
        .layers
        .iter()
        .enumerate()
        .filter(|(_, kind)| kind.routed)
        .flat_map(|(l, _)| [names::ffn_gate_exps(l), names::ffn_up_exps(l)])
        .collect())
}

/// The real DSpark draft: `$BLOOMERY_DSPARK_MODEL`.
fn default_draft_source() -> Option<String> {
    std::env::var("BLOOMERY_DSPARK_MODEL").ok()
}

/// The coverage table of the target's architecture keys; `None` is a key
/// the map does not cover.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 37] = [
        "context_length",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "rope.scaling.type",
        "rope.scaling.factor",
        "rope.scaling.original_context_length",
        "rope.scaling.yarn_beta_fast",
        "rope.scaling.yarn_beta_slow",
        "rope.freq_base",
        "attention.layer_norm_rms_epsilon",
        "expert_count",
        "expert_used_count",
        "expert_gating_func",
        "attention.key_length",
        "attention.value_length",
        "rope.dimension_count",
        "attention.q_lora_rank",
        "attention.sliding_window",
        "expert_shared_count",
        "expert_weights_scale",
        "expert_weights_norm",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
        "attention.output_group_count",
        "attention.output_lora_rank",
        "attention.compress_rope_freq_base",
        "hyper_connection.count",
        "hyper_connection.sinkhorn_iterations",
        "hyper_connection.epsilon",
        "engram.head_count",
        "engram.key_length",
        "engram.max_ngram_size",
        "engram.multipliers",
        "engram.token_map",
        "engram.pad_id",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "expert_feed_forward_length" => KeyRule::Ff,
        "attention.compress_ratios" => KeyRule::Ratios,
        "swiglu_clamp_exp" | "swiglu_clamp_shexp" => KeyRule::PerLayer,
        "engram.layer_ids" => KeyRule::LayerIds,
        "engram.primes" | "engram.offsets" => KeyRule::Table,
        "hash_layer_count" => KeyRule::Zero("hash-routed layers are not mapped"),
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of a 1-D F32 tensor, by its name without the `blk.N.` prefix:
/// gains 1, sinks, the text router bias and hyper-connection bases 0,
/// hyper-connection scales 1. `exp_probs_b_vl` is not one: [`spread_value`].
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 14] = [
        ("attn_norm.weight", 1.0),
        ("ffn_norm.weight", 1.0),
        ("attn_q_a_norm.weight", 1.0),
        ("attn_kv_a_norm.weight", 1.0),
        ("attn_compressor_norm.weight", 1.0),
        ("indexer.k_norm.weight", 1.0),
        ("output_norm.weight", 1.0),
        ("enc.output_norm.weight", 1.0),
        ("attn_sinks.weight", 0.0),
        ("exp_probs_b.bias", 0.0),
        ("hc_attn_base.weight", 0.0),
        ("hc_ffn_base.weight", 0.0),
        ("hc_attn_scale.weight", 1.0),
        ("hc_ffn_scale.weight", 1.0),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The half-width of a 1-D F32 tensor drawn uniformly, by its name without the
/// `blk.N.` prefix: the media router bias, [`VL_BIAS_HALF_WIDTH`].
pub fn spread_value(leaf: &str) -> Option<f32> {
    (leaf == "exp_probs_b_vl.bias").then_some(VL_BIAS_HALF_WIDTH)
}

/// V4.1's rules.
struct V41;

impl Family for V41 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// The routed stacks and the shared expert, which the chain sizes at the
    /// routed ff: gate and up carry it in their second dim, down in its first.
    fn ff_axis(&self, leaf: &str) -> Option<usize> {
        match leaf {
            "ffn_gate_exps.weight"
            | "ffn_up_exps.weight"
            | "ffn_gate_shexp.weight"
            | "ffn_up_shexp.weight" => Some(1),
            "ffn_down_exps.weight" | "ffn_down_shexp.weight" => Some(0),
            _ => None,
        }
    }

    fn tables(&self, source: &Split, _spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        Ok(Box::new(Engram::read(source)?))
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

/// The first `n` primes above `floor`.
fn primes_above(floor: u64, n: usize) -> Vec<u64> {
    let is_prime = |p: u64| {
        p >= 2
            && (2..)
                .take_while(|d| d * d <= p)
                .all(|d| !p.is_multiple_of(d))
    };
    (floor + 1..).filter(|&p| is_prime(p)).take(n).collect()
}

/// The source's engram layout: per site its buckets' primes; the offsets
/// must be each site's exclusive prefix sums, as the fixture writes them.
struct EngramSource {
    layer_ids: Vec<usize>,
    cols: usize,
    primes: Vec<u64>,
    prime_tag: Value,
    offset_tag: Value,
}

fn engram_source(source: &Split) -> Result<EngramSource, FixtureError> {
    let key = |s: &str| format!("{TARGET_ARCH}.engram.{s}");
    let get = |s: &str| {
        source
            .value(&key(s))
            .ok_or_else(|| meta(&key(s), "is absent"))
    };
    let layer_ids = items(&key("layer_ids"), get("layer_ids")?)?
        .iter()
        .map(|v| unsigned(&key("layer_ids"), v).map(|l| l as usize))
        .collect::<Result<Vec<_>, _>>()?;
    if layer_ids.is_empty() {
        return Err(meta(&key("layer_ids"), "is empty"));
    }
    let heads = unsigned(&key("head_count"), get("head_count")?)? as usize;
    let ngram = unsigned(&key("max_ngram_size"), get("max_ngram_size")?)? as usize;
    let cols = ngram
        .checked_sub(1)
        .map(|n| n * heads)
        .filter(|&c| c > 0)
        .ok_or_else(|| meta(&key("max_ngram_size"), "leaves no bucket"))?;
    let p = items(&key("primes"), get("primes")?)?;
    let o = items(&key("offsets"), get("offsets")?)?;
    let want = layer_ids.len() * cols;
    if p.len() != want || o.len() != want {
        return Err(meta(
            &key("primes"),
            format!(
                "{} primes and {} offsets for {want} buckets",
                p.len(),
                o.len()
            ),
        ));
    }
    let primes = p
        .iter()
        .map(|v| unsigned(&key("primes"), v))
        .collect::<Result<Vec<_>, _>>()?;
    for (site, (ps, os)) in primes.chunks(cols).zip(o.chunks(cols)).enumerate() {
        let mut acc = 0u64;
        for (b, (&pr, ov)) in ps.iter().zip(os).enumerate() {
            if unsigned(&key("offsets"), ov)? != acc {
                return Err(meta(
                    &key("offsets"),
                    format!("site {site} bucket {b} is not its site's prefix sum {acc}"),
                ));
            }
            acc += pr;
        }
    }
    Ok(EngramSource {
        layer_ids,
        cols,
        primes,
        prime_tag: p[0].clone(),
        offset_tag: o[0].clone(),
    })
}

/// The engram tables of one plan: the fixture's primes, and each site's rows
/// in the fixture and in the source.
struct Engram {
    source: EngramSource,
    fx_primes: Vec<u64>,
    fx_rows: Vec<u64>,
    src_rows: Vec<u64>,
}

impl Engram {
    fn read(source: &Split) -> Result<Engram, FixtureError> {
        let source = engram_source(source)?;
        let fx_primes = primes_above(ENGRAM_PRIME_FLOOR, source.primes.len());
        let site_rows = |p: &[u64]| {
            p.chunks(source.cols)
                .map(|c| c.iter().sum::<u64>())
                .collect::<Vec<u64>>()
        };
        let (fx_rows, src_rows) = (site_rows(&fx_primes), site_rows(&source.primes));
        Ok(Engram {
            source,
            fx_primes,
            fx_rows,
            src_rows,
        })
    }
}

impl Tables for Engram {
    fn key(&self, key: &str, suffix: &str, _v: &Value) -> Result<Value, FixtureError> {
        match suffix {
            "engram.primes" => Ok(Value::Array(
                self.fx_primes
                    .iter()
                    .map(|&p| int_like(key, &self.source.prime_tag, p))
                    .collect::<Result<_, _>>()?,
            )),
            "engram.offsets" => {
                let mut out = Vec::with_capacity(self.fx_primes.len());
                for site in self.fx_primes.chunks(self.source.cols) {
                    let mut acc = 0u64;
                    for &p in site {
                        out.push(int_like(key, &self.source.offset_tag, acc)?);
                        acc += p;
                    }
                }
                Ok(Value::Array(out))
            }
            _ => Err(meta(key, "is a table key the engram tables do not compute")),
        }
    }

    fn dims(
        &self,
        name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        let Some(l) = layer.filter(|_| leaf == "engram_embd.weight") else {
            return Ok(None);
        };
        let bad = |detail: String| FixtureError::Tensor {
            name: name.to_string(),
            detail,
        };
        let site = self
            .source
            .layer_ids
            .iter()
            .position(|&x| x == l)
            .ok_or_else(|| bad("is the table of a layer engram.layer_ids does not list".into()))?;
        if dims.len() != 2 || dims[1] != self.src_rows[site] {
            let want = self.src_rows[site];
            return Err(bad(format!(
                "has dims {dims:?}; its site's primes sum to {want}"
            )));
        }
        Ok(Some(vec![dims[0], self.fx_rows[site]]))
    }

    fn lines(&self) -> Vec<String> {
        self.fx_rows
            .iter()
            .zip(&self.src_rows)
            .enumerate()
            .map(|(s, (fx, src))| format!("engram site {s} rows={fx} (source {src})"))
            .collect()
    }
}

/// Each engram site's rows in the fixture and in `source`, in site order.
pub fn engram_rows(source: &Split) -> Result<Vec<(u64, u64)>, FixtureError> {
    let e = Engram::read(source)?;
    Ok(e.fx_rows.into_iter().zip(e.src_rows).collect())
}

/// The last `k` of `n` layers; `None` when `k > n`.
fn last_layers(n: usize, k: usize) -> Option<std::ops::Range<usize>> {
    n.checked_sub(k).map(|first| first..n)
}

/// The DSpark draft's rules.
struct DSpark;

impl DraftRules for DSpark {
    /// The real draft's metadata with `target_layers` moved to the fixture's
    /// last layers (it must name the source's last ones) and, under an ff
    /// override, its ff moved with the target's (it must be the target's).
    fn kvs(
        &self,
        draft: &Split,
        n_source: usize,
        n_fixture: usize,
        ff: Option<(u64, u64)>,
    ) -> Result<Kvs, FixtureError> {
        let tl_key = draft.arch_key("target_layers");
        let ff_key = draft.arch_key("expert_feed_forward_length");
        let mut kvs = Vec::new();
        for (k, v) in draft.iter_kv() {
            if let (Some((from, to)), true) = (ff, k == ff_key) {
                let got = unsigned(k, v)?;
                if got != from {
                    return Err(meta(
                        k,
                        format!("is {got}, not the target's ff {from}: one override moves both"),
                    ));
                }
                kvs.push((k.to_string(), int_like(k, v, to)?));
                continue;
            }
            if k != tl_key {
                kvs.push((k.to_string(), v.clone()));
                continue;
            }
            let a = items(k, v)?;
            let got = a
                .iter()
                .map(|x| unsigned(k, x).map(|l| l as usize))
                .collect::<Result<Vec<_>, _>>()?;
            let last: Vec<usize> = last_layers(n_source, got.len())
                .ok_or_else(|| {
                    meta(
                        k,
                        format!("names {} layers of a {n_source}-layer source", got.len()),
                    )
                })?
                .collect();
            if got.is_empty() || got != last {
                return Err(meta(
                    k,
                    format!("is {got:?}, not the source's last layers {last:?}"),
                ));
            }
            let tag = &a[0];
            let moved = last_layers(n_fixture, got.len())
                .ok_or_else(|| {
                    meta(
                        k,
                        format!(
                            "names {} layers, more target layers than the fixture's {n_fixture}",
                            got.len()
                        ),
                    )
                })?
                .map(|f| int_like(k, tag, f as u64))
                .collect::<Result<_, _>>()?;
            kvs.push((k.to_string(), Value::Array(moved)));
        }
        if !kvs.iter().any(|(k, _)| *k == tl_key) {
            return Err(meta(&tl_key, "is absent"));
        }
        if ff.is_some() && !kvs.iter().any(|(k, _)| *k == ff_key) {
            return Err(meta(&ff_key, "is absent"));
        }
        Ok(kvs)
    }

    fn check(
        &self,
        spec: &FixtureSpec,
        draft: &Split,
        target: &Split,
        whole: bool,
    ) -> Result<(), FixtureError> {
        check_draft(spec, draft, target, whole).map(|_| ())
    }
}

/// Source layer `l`'s index in the fixture of `layers`.
fn fixture_layer(layers: &[usize], l: usize, what: &str) -> Result<usize, FixtureError> {
    layers
        .iter()
        .position(|&m| m == l)
        .ok_or_else(|| FixtureError::Mismatch {
            what: what.to_string(),
            detail: format!("reads source layer {l}, which the map does not hold"),
        })
}

/// Source kind `k` with its layer references moved through `layers`.
fn remap_kind(layers: &[usize], k: &LayerKind, what: &str) -> Result<LayerKind, FixtureError> {
    let mut out = *k;
    if let Some(s) = k.stream {
        out.stream = Some(Stream {
            ratio: s.ratio,
            kv_source: fixture_layer(layers, s.kv_source, what)?,
            index_key_source: fixture_layer(layers, s.index_key_source, what)?,
            topk_source: fixture_layer(layers, s.topk_source, what)?,
        });
    }
    if let Some(d) = k.dense {
        out.dense = Some(DenseStream {
            ratio: d.ratio,
            kv_source: fixture_layer(layers, d.kv_source, what)?,
        });
    }
    Ok(out)
}

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds are
/// the source layers' kinds, their layer references moved through the map.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    if fx.n_layer != spec.layers.len() {
        return Err(FixtureError::Mismatch {
            what: "layer count".into(),
            detail: format!("{}, not {}", fx.n_layer, spec.layers.len()),
        });
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        let want = remap_kind(&spec.layers, &src.layers[l], &what)?;
        if fx.layers[f] != want {
            return Err(FixtureError::Mismatch {
                what,
                detail: format!("kind {:?}, the source's {want:?}", fx.layers[f]),
            });
        }
    }
    let want_ff = spec.ff.map_or(src.experts.ff, |f| f as usize);
    let want = Experts {
        ff: want_ff,
        ..src.experts
    };
    if fx.experts != want {
        return Err(FixtureError::Mismatch {
            what: "the experts".into(),
            detail: format!("{:?}, the source's at ff {want_ff}: {want:?}", fx.experts),
        });
    }
    check_biases(fixture, &fx)?;
    Ok(fx)
}

/// The f32 values of tensor `name` of `fixture`.
fn f32_tensor(fixture: &Split, name: &str) -> Result<Vec<f32>, FixtureError> {
    let bad = |detail: String| FixtureError::Tensor {
        name: name.to_string(),
        detail,
    };
    let (s, info) = fixture
        .find(name)
        .ok_or_else(|| bad("is not in the fixture".into()))?;
    if info.ty != GgmlType::F32 {
        return Err(bad(format!("is {}, not F32", info.ty)));
    }
    let shard = fixture
        .shard(s)
        .ok_or_else(|| bad("has no shard".to_string()))?;
    let bytes = shard.data(info)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Every routed layer's media bias differs from its text bias between experts
/// by at least [`BIAS_SPREAD_MIN`]: a difference the same on every expert (zero
/// among them, the two biases equal) moves no router pick, and the media
/// gate's coverage clause needs `bias_vl`'s picks to differ from `bias`'s at
/// an image and a separator position.
fn check_biases(fixture: &Split, hp: &Hparams) -> Result<(), FixtureError> {
    for (l, _) in hp.layers.iter().enumerate().filter(|(_, k)| k.routed) {
        let (text, media) = (
            f32_tensor(fixture, &names::exp_probs_b(l))?,
            f32_tensor(fixture, &names::exp_probs_b_vl(l))?,
        );
        if text.len() != media.len() {
            return Err(FixtureError::Mismatch {
                what: format!("layer {l}'s router biases"),
                detail: format!("{} text and {} media values", text.len(), media.len()),
            });
        }
        let (lo, hi) = text
            .iter()
            .zip(&media)
            .map(|(t, m)| m - t)
            .fold((f32::MAX, f32::MIN), |(lo, hi), d| (lo.min(d), hi.max(d)));
        let span = hi - lo;
        if span.is_nan() || span < BIAS_SPREAD_MIN {
            return Err(FixtureError::Mismatch {
                what: format!("layer {l}'s router biases"),
                detail: format!(
                    "exp_probs_b_vl − exp_probs_b spans [{lo}, {hi}] over the experts, under \
                     {BIAS_SPREAD_MIN}: the media picks cannot differ from the text picks (the \
                     media gate's coverage clause refuses itself)"
                ),
            });
        }
    }
    Ok(())
}

/// The draft fixture reads as a draft whose target layers are the fixture's
/// last ones; with both files whole, its inventory against `target` holds.
pub fn check_draft(
    spec: &FixtureSpec,
    draft: &Split,
    target: &Split,
    whole: bool,
) -> Result<DraftHparams, FixtureError> {
    let hp = DraftHparams::read(draft)?;
    let n = spec.layers.len();
    let want: Vec<usize> = last_layers(n, hp.target_layers.len())
        .ok_or_else(|| FixtureError::Mismatch {
            what: "draft target_layers".into(),
            detail: format!(
                "{:?} is more target layers than the fixture's {n}",
                hp.target_layers
            ),
        })?
        .collect();
    if hp.target_layers != want {
        return Err(FixtureError::Mismatch {
            what: "draft target_layers".into(),
            detail: format!("{:?}, not {want:?}", hp.target_layers),
        });
    }
    if whole {
        dspark::inventory(draft, &hp, target)?.check()?;
    }
    Ok(hp)
}
