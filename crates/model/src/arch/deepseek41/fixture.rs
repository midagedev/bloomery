//! The V4.1 gate fixture's spec ([`spec`]): the facts of a V4.1 file the
//! common generator (`crate::fixture`) cannot know.
//!
//! The fixture is nine layers, one per layer kind the step distinguishes
//! ([`LAYER_MAP`], whose compress ratios must read [`FIXTURE_RATIOS`]), and
//! the source's globals. Every block's `d` (and `dmin`) lies in
//! [[`D_MIN`], [`D_MAX`]]. Every key of the coverage table ([`key_rule`]) keeps
//! its source's value but the layer count, the per-layer arrays, the engram
//! layer ids, and the engram primes and offsets: an engram table's rows
//! follow the fixture's own hash primes, the smallest above
//! [`ENGRAM_PRIME_FLOOR`], so a table is a few hundred thousand rows instead
//! of several hundred million. A 1-D F32 tensor holds its constant
//! ([`const_value`]): gains 1, sinks, biases and hyper-connection bases 0,
//! hyper-connection scales 1.
//!
//! The DSpark draft fixture ([`DRAFT_FILE`]) is the real draft's tensors at
//! their shapes with random weights by the same rules, `target_layers` moved
//! to the fixture's last layers, and the generator's five keys.

use gguf::{Split, Value};

use super::hparams::{DenseStream, Hparams, LayerKind, Stream};
use crate::arch::dspark::{self, DraftHparams};
use crate::fixture::{
    CardBudget, DraftRules, DraftSpec, Family, FixtureError, FixtureSpec, KeyRule, Kvs, Tables,
    Window, int_like, items, meta, unsigned,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]`: one layer per kind
/// the step's launch table tells apart (window, window+engram, the r2 source
/// and reader, the r2 source with engram, the r1 source that ends the
/// compressed layers, the r1 readers with and without an indexer), and a
/// last reader whose top-k source is not its kv source.
pub const LAYER_MAP: [usize; 9] = [0, 1, 2, 3, 14, 20, 21, 24, 25];

/// The compress ratios the map must read from the source: the check that
/// the map still picks the kinds it was chosen for.
pub const FIXTURE_RATIOS: [u64; 9] = [0, 0, 2, 2, 2, 1, 1, 1, 1];

/// The fixture design's d window, [2^-13, 2^-10]: normal f16 values that
/// every V4.1 type's scale fits at every K the file holds.
pub const D_MIN: f32 = 1.0 / 8192.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 1024.0;

/// The card budget the fixture design derived for the gate placement:
/// 6,580 MiB keeps about as many experts a layer as the real gate placement.
pub const DEFAULT_CARD_BUDGET: u64 = 6580 << 20;

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

/// The V4.1 fixture: [`LAYER_MAP`], [`FIXTURE_RATIOS`], [`DEFAULT_CARD_BUDGET`],
/// the window [[`D_MIN`], [`D_MAX`]], the real ff, the V4.1 file as the
/// default source, and the DSpark draft.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: TARGET_ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: FIXTURE_RATIOS.to_vec(),
        card_budget: CardBudget::Fixed(DEFAULT_CARD_BUDGET),
        window: Window::new(D_MIN, D_MAX).expect("[2^-13, 2^-10] is inside the normal f16 values"),
        ff: None,
        default_source: gguf::v41::model,
        draft: Some(DraftSpec {
            arch: crate::arch::DFLASH,
            file: DRAFT_FILE,
            default_source: default_draft_source,
            rules: &DSpark,
        }),
        family: &V41,
    }
}

/// The real DSpark draft: `$BLOOMERY_DSPARK_MODEL`.
fn default_draft_source() -> Option<String> {
    std::env::var("BLOOMERY_DSPARK_MODEL").ok()
}

/// The coverage table of the target's architecture keys; `None` is a key
/// the map does not cover.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 38] = [
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
        "expert_feed_forward_length",
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
/// gains 1, sinks, biases and hyper-connection bases 0, hyper-connection
/// scales 1.
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 15] = [
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
        ("exp_probs_b_vl.bias", 0.0),
        ("hc_attn_base.weight", 0.0),
        ("hc_ffn_base.weight", 0.0),
        ("hc_attn_scale.weight", 1.0),
        ("hc_ffn_scale.weight", 1.0),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
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
    /// last layers: it must name the source's last ones.
    fn kvs(&self, draft: &Split, n_source: usize, n_fixture: usize) -> Result<Kvs, FixtureError> {
        let tl_key = draft.arch_key("target_layers");
        let mut kvs = Vec::new();
        for (k, v) in draft.iter_kv() {
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
    Ok(fx)
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
