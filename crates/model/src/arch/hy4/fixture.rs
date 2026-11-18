//! The Hy4-flash gate fixture's spec ([`spec`]): the facts of a `hy_v4_flash` file the common
//! generator (`crate::fixture`) cannot know. The model is 42 layers of gated latent attention (a
//! sparse layer with an indexer every fourth, a window layer between) under four-stream
//! hyper-connections, a dense first layer over routed latent experts, and four engram sites before
//! the mixer; the file is a two-shard split whose second shard holds the engram table alone.
//!
//! The fixture is five real layers, renamed `blk.L.` → `blk.f.`, and the next-token block dropped:
//!
//! | fixture layer | source layer | what it is |
//! |---|---|---|
//! | f0 | L0 | sparse attention, the dense feed-forward |
//! | f1 | L1 | sparse attention, an engram site, routed experts, a Q6_K down stack |
//! | f2 | L6 | window attention, routed experts, a Q4_K down stack |
//! | f3 | L11 | window attention, an engram site, routed experts, a Q4_K down stack |
//! | f4 | L41 | the last sparse layer, routed experts, a Q6_K down stack |
//!
//! Every kind of the real file's layers is one of these (both mixers, both head counts and rope
//! bases, a site on each mixer, both down types), read from the source's own arrays and tensor
//! types ([`KINDS`]), never from a rule. What it keeps, because the card kernels are built for it:
//! the hidden (3,072), the head counts (32 sparse, 64 window), the latent geometry (512 + 64, value
//! width 256), the indexer (16 heads of 128, pools of 4), the four hyper-connection streams, the
//! 512 routed experts and the 16 picked, the experts' latent (1,536), the engram's 8 heads an order
//! over 3-grams of 96-wide rows, the real vocabulary and tokenizer, and every tensor's type.
//!
//! What it minifies, each by its key: the routed experts' width and the shared expert's (1,536 →
//! [`FIXTURE_FF`], a whole block of both K-quants), the dense layer's (13,824 → [`DENSE_FF`]), the
//! window (512 → [`WINDOW`]) and the indexer's selection (2,048 → [`TOP_K`], pools of 4: 15 closed
//! groups and the open one), and the engram table: its sites are the real file's first two, each
//! head's bucket is the next prime above [`ENGRAM_PRIME_FLOOR`] (the real file's are ten million
//! rows), the buckets follow one another in one table, as in the real file.
//!
//! The table's rows are scale-less F8_E4M3 (id 43, one byte a value): random codes, since the real
//! file's rows are scaled to the format's amax and the scale dropped, so the first norm that reads
//! a row undoes it.
//!
//! The fills (the generator's matrix rule is σ 1/√K, so a projection of a unit-RMS input has unit
//! RMS): the norm gains are 1, the hyper-connection bases 0 and scales 1, the engram's pre-gain 1;
//! the attention sinks and the router's selection bias are spreads ([`SINK_HALF_WIDTH`],
//! [`BIAS_HALF_WIDTH`]), per element, so a body that broadcasts one value or skips one fails and
//! the router's top-16 is not a set of ties.
//!
//! The card budget is fixed, not planned: the architecture has no placement yet, so no plan of the
//! written file chooses a cap; the recorded value keeps the whole fixture resident. No draft file
//! and no r8 sidecar.

use gguf::{GgmlType, Split, Value};

use super::hparams::Hparams;
use super::names::ENGRAM_TABLE;
use crate::arch::names::blk;
use crate::fixture::{
    CardBudget, Family, FixtureError, FixtureSpec, KeyRule, Tables, Window, int_like, items, meta,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 5] = [0, 1, 6, 11, 41];

/// The routed experts' and the shared expert's width the fixture writes in place of the source's
/// 1,536: `expert_feed_forward_length`. A whole block of Q4_K and Q6_K, the K of the down stacks.
pub const FIXTURE_FF: u64 = 256;

/// The dense layer's width the fixture writes: `feed_forward_length`, from 13,824.
pub const DENSE_FF: u64 = 1024;

/// The window layers' positions the fixture writes: `attention.sliding_window`, from 512.
pub const WINDOW: u64 = 64;

/// The tokens a sparse layer attends the fixture writes: `attention.indexer.top_k`, from 2,048;
/// with pools of 4 that is 15 closed groups and the open one.
pub const TOP_K: u64 = 64;

/// The engram buckets' primes are the smallest above this, as many as the kept sites' buckets, in
/// site order.
pub const ENGRAM_PRIME_FLOOR: u64 = 1 << 14;

/// The d window, [2^-14, 2^-4]: normal f16 values every type's scale fits at every K the fixture
/// holds (256 to 16,384).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 16.0;

/// The most tensor data a shard holds: the shared default, above the fixture's tensor data, so the
/// fixture is one shard.
pub const SHARD_BYTES: u64 = crate::fixture::DEFAULT_SHARD_BYTES;

/// The card byte budget the fixture records. Above the fixture's tensor data, it keeps the whole
/// model resident; the architecture has no placement to plan a cap through.
pub const CARD_BUDGET: u64 = 4 << 30;

/// The half-width of the attention sinks (`attn_sinks`), drawn per head: a sink of order one
/// against logits of order one, so it moves every head's softmax and none is a constant.
pub const SINK_HALF_WIDTH: f32 = 0.5;

/// The half-width of the router's selection bias (`exp_probs_b.bias`), drawn per expert: wide
/// enough that it steers the pick, narrow enough that the sigmoid scores still decide it.
pub const BIAS_HALF_WIDTH: f32 = 0.05;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "hy4-fixture";

/// The real first shard `verify` opens when none is named: the profile's file.
pub const DEFAULT_MODEL: &str = "/models/Hy4-flash-GGUF/Hy4-flash-Q4_K_M-00001-of-00002.gguf";

const ARCH: &str = super::ARCH;

/// What a source layer must read as for the map to keep a fixture layer's kind: the rows of the
/// table above.
struct Layer {
    /// The layer runs sparse attention (32 heads, the indexer), not a window.
    dsa: bool,
    /// The layer is the dense lead.
    dense: bool,
    /// The layer is an engram site.
    site: bool,
    /// The type of the routed down stack; `None` for the dense layer.
    down: Option<GgmlType>,
}

/// The five kinds of [`LAYER_MAP`], layer by layer.
const KINDS: [Layer; 5] = [
    Layer {
        dsa: true,
        dense: true,
        site: false,
        down: None,
    },
    Layer {
        dsa: true,
        dense: false,
        site: true,
        down: Some(GgmlType::Q6_K),
    },
    Layer {
        dsa: false,
        dense: false,
        site: false,
        down: Some(GgmlType::Q4_K),
    },
    Layer {
        dsa: false,
        dense: false,
        site: true,
        down: Some(GgmlType::Q4_K),
    },
    Layer {
        dsa: true,
        dense: false,
        site: false,
        down: Some(GgmlType::Q6_K),
    },
];

/// The Hy4-flash fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], [`FIXTURE_FF`], the
/// profile's file as the default source, no draft, no sidecar and a fixed card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Fixed(CARD_BUDGET),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-4] is inside the normal f16 values"),
        ff: Some(FIXTURE_FF),
        shard_bytes: SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Hy4,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. What the
/// fixture resizes is a [`KeyRule::Set`] or [`KeyRule::Ff`]; the per-layer arrays follow the map
/// ([`KeyRule::PerLayer`], and the indexer's, which stops at the trunk, and the engram's sites and
/// buckets are [`KeyRule::Table`]); the rest is copied.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: &[&str] = &[
        "context_length",
        "embedding_length",
        "vocab_size",
        "leading_dense_block_count",
        "attention.layer_norm_rms_epsilon",
        "attention.key_length",
        "attention.value_length",
        "attention.key_length_mla",
        "attention.value_length_mla",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.kpool",
        "attention.indexer.kpool_select_tail",
        "rope.dimension_count",
        "rope.freq_base",
        "rope.freq_base_swa",
        "expert_count",
        "expert_used_count",
        "expert_shared_count",
        "expert_latent_length",
        "expert_weights_scale",
        "expert_weights_norm",
        "expert_gating_func",
        "hyper_connection.count",
        "hyper_connection.epsilon",
        "hyper_connection.magnitude",
        "ple.ngram_size",
        "ple.heads_per_ngram",
        "embedding_length_per_layer_input",
        "engram.pad_token_id",
        "engram.token_map",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "nextn_predict_layers" => KeyRule::Set(0),
        "feed_forward_length" => KeyRule::Set(DENSE_FF),
        "expert_feed_forward_length" => KeyRule::Ff,
        "attention.sliding_window" => KeyRule::Set(WINDOW),
        "attention.indexer.top_k" => KeyRule::Set(TOP_K),
        "attention.head_count"
        | "attention.head_count_kv"
        | "attention.sliding_window_pattern"
        | "swiglu_clamp_exp" => KeyRule::PerLayer,
        "attention.indexer.types"
        | "ple.layers"
        | "ple.layer_multipliers"
        | "ple.head_offsets"
        | "ple.head_vocab_sizes" => KeyRule::Table,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.` (the whole name for a
/// model-level one): the norm gains 1, the hyper-connection bases 0 and scales 1, the engram's
/// pre-gain and lookup gains 1. The attention sinks and the selection bias are spreads
/// ([`spread_value`]).
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 17] = [
        ("attn_norm.weight", 1.0),
        ("ffn_norm.weight", 1.0),
        ("attn_q_a_norm.weight", 1.0),
        ("attn_kv_a_norm.weight", 1.0),
        ("post_attention_norm.weight", 1.0),
        ("post_ffw_norm.weight", 1.0),
        ("indexer.k_norm.weight", 1.0),
        ("indexer_compressor_norm.weight", 1.0),
        ("engram_pre_gamma.weight", 1.0),
        ("engram_lookup_norm.weight", 1.0),
        ("hc_attn_base.weight", 0.0),
        ("hc_ffn_base.weight", 0.0),
        ("hc_attn_scale.weight", 1.0),
        ("hc_ffn_scale.weight", 1.0),
        ("output_norm.weight", 1.0),
        ("output_hc_base.weight", 0.0),
        ("output_hc_scale.weight", 1.0),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The spread of an F32 tensor the family names: the attention sinks ([`SINK_HALF_WIDTH`]) and the
/// router's selection bias ([`BIAS_HALF_WIDTH`]).
pub fn spread_value(leaf: &str) -> Option<f32> {
    match leaf {
        "attn_sinks.weight" => Some(SINK_HALF_WIDTH),
        "exp_probs_b.bias" => Some(BIAS_HALF_WIDTH),
        _ => None,
    }
}

/// Hy4-flash's rules.
struct Hy4;

impl Family for Hy4 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// The keys a [`KeyRule::Set`] rewrites, which the plan walks only if the source has them: a
    /// source without one is a file whose header the fixture would not match.
    fn required_keys(&self) -> &'static [&'static str] {
        &[
            "nextn_predict_layers",
            "feed_forward_length",
            "attention.sliding_window",
            "attention.indexer.top_k",
        ]
    }

    /// The routed stacks and the shared expert, which both run at the routed ff: gate and up carry
    /// it in their second dim, down in its first. The dense layer's width is the tables'.
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

    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        Ok(Box::new(Cut::read(source, &spec.layers)?))
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

/// The map's layers must read as [`KINDS`] in `src`, layer by layer from the source's own arrays
/// and tensor types. A source that differs is refused by name, so a rule of the model that moved
/// cannot hide behind a map that still has the right layer numbers.
fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    if layers.len() != KINDS.len() {
        return Err(FixtureError::Mismatch {
            what: "layer map".into(),
            detail: format!(
                "holds {} layers, the fixture is built for the {} of the kinds above",
                layers.len(),
                KINDS.len()
            ),
        });
    }
    for (f, (&l, kind)) in layers.iter().zip(&KINDS).enumerate() {
        if l >= src.n_layer {
            return Err(FixtureError::NoSourceLayer {
                layer: l,
                n_layer: src.n_layer,
            });
        }
        let bad = |got: String, want: String| FixtureError::Mismatch {
            what: "layer map".into(),
            detail: format!("source layer {l} (fixture {f}) {got}, the kind keeps it {want}"),
        };
        let mixer = |dsa: bool| {
            if dsa {
                "a sparse-attention layer"
            } else {
                "a window layer"
            }
        };
        if src.dsa(l) != kind.dsa {
            return Err(bad(
                format!("is {}", mixer(src.dsa(l))),
                mixer(kind.dsa).into(),
            ));
        }
        let ffn = |dense: bool| if dense { "dense" } else { "routed" };
        if src.dense(l) != kind.dense {
            return Err(bad(
                format!("is {}", ffn(src.dense(l))),
                ffn(kind.dense).into(),
            ));
        }
        let site = |on: bool| {
            if on {
                "an engram site"
            } else {
                "no engram site"
            }
        };
        if src.site(l) != kind.site {
            return Err(bad(
                format!("is {}", site(src.site(l))),
                site(kind.site).into(),
            ));
        }
        let got = source
            .find(&blk(l, "ffn_down_exps.weight"))
            .map(|(_, t)| t.ty);
        if got != kind.down {
            return Err(bad(
                format!("holds ffn_down_exps.weight as {got:?}"),
                format!("as {:?}", kind.down),
            ));
        }
    }
    Ok(())
}

/// The engram table of one plan: the kept sites and their buckets.
struct Cut {
    /// The source's layer map, fixture layer by fixture layer.
    layers: Vec<usize>,
    /// The source's sites in module order, and each one's fixture layer when the map holds it.
    sites: Vec<Option<usize>>,
    /// Multipliers a site: `ple.ngram_size`.
    ngram: usize,
    /// The fixture's buckets: the primes above [`ENGRAM_PRIME_FLOOR`], the kept sites' columns.
    primes: Vec<u64>,
    /// The table's rows in the source and in the fixture.
    src_rows: u64,
    fx_rows: u64,
    /// The source's dense width, `feed_forward_length`.
    dense_ff: u64,
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

impl Cut {
    fn read(source: &Split, layers: &[usize]) -> Result<Cut, FixtureError> {
        let src = Hparams::read(source)?;
        check_map(&src, source, layers)?;
        let eng = src
            .engram
            .as_ref()
            .ok_or_else(|| meta("ple.layers", "is absent from the source: no engram site"))?;
        let sites: Vec<Option<usize>> = eng
            .layers
            .iter()
            .map(|s| layers.iter().position(|l| l == s))
            .collect();
        let kept = sites.iter().flatten().count();
        if kept == 0 {
            return Err(meta(
                "ple.layers",
                "lists no site the map holds: the fixture runs no engram",
            ));
        }
        let ngram = eng.ngram;
        let primes = primes_above(ENGRAM_PRIME_FLOOR, kept * eng.n_cols());
        let fx_rows = primes.iter().sum();
        Ok(Cut {
            layers: layers.to_vec(),
            sites,
            ngram,
            primes,
            src_rows: eng.rows,
            fx_rows,
            dense_ff: src.dense_ff as u64,
        })
    }

    /// The source modules the map holds, in module order.
    fn kept(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.sites
            .iter()
            .enumerate()
            .filter_map(|(e, f)| f.map(|f| (e, f)))
    }
}

impl Tables for Cut {
    fn key(&self, key: &str, suffix: &str, v: &Value) -> Result<Value, FixtureError> {
        let tag = |v: &Value| -> Result<Value, FixtureError> {
            items(key, v)?
                .first()
                .cloned()
                .ok_or_else(|| meta(key, "is empty"))
        };
        match suffix {
            "attention.indexer.types" => {
                let a = items(key, v)?;
                let top = self.layers.iter().max().copied().unwrap_or(0);
                if a.len() <= top {
                    return Err(meta(
                        key,
                        format!("has {} values, the map reaches layer {top}", a.len()),
                    ));
                }
                Ok(Value::Array(
                    self.layers.iter().map(|&l| a[l].clone()).collect(),
                ))
            }
            "ple.layers" => {
                let tag = tag(v)?;
                Ok(Value::Array(
                    self.kept()
                        .map(|(_, f)| int_like(key, &tag, f as u64))
                        .collect::<Result<_, _>>()?,
                ))
            }
            "ple.layer_multipliers" => {
                let a = items(key, v)?;
                if a.len() != self.sites.len() * self.ngram {
                    return Err(meta(
                        key,
                        format!(
                            "has {} values for {} sites of {} orders",
                            a.len(),
                            self.sites.len(),
                            self.ngram
                        ),
                    ));
                }
                Ok(Value::Array(
                    self.kept()
                        .flat_map(|(e, _)| a[e * self.ngram..(e + 1) * self.ngram].iter().cloned())
                        .collect(),
                ))
            }
            "ple.head_vocab_sizes" => {
                let tag = tag(v)?;
                Ok(Value::Array(
                    self.primes
                        .iter()
                        .map(|&p| int_like(key, &tag, p))
                        .collect::<Result<_, _>>()?,
                ))
            }
            "ple.head_offsets" => {
                let tag = tag(v)?;
                let mut acc = 0u64;
                let mut out = Vec::with_capacity(self.primes.len());
                for &p in &self.primes {
                    out.push(int_like(key, &tag, acc)?);
                    acc += p;
                }
                Ok(Value::Array(out))
            }
            _ => Err(meta(
                key,
                "is a table key the Hy4-flash tables do not compute",
            )),
        }
    }

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
        if layer.is_none() && leaf == ENGRAM_TABLE {
            if dims.len() != 2 || dims[1] != self.src_rows {
                return Err(bad(format!(
                    "has dims {dims:?}; the source's buckets sum to {} rows",
                    self.src_rows
                )));
            }
            return Ok(Some(vec![dims[0], self.fx_rows]));
        }
        let axis = match leaf {
            "ffn_gate.weight" | "ffn_up.weight" => 1,
            "ffn_down.weight" => 0,
            _ => return Ok(None),
        };
        if dims.len() != 2 || dims[axis] != self.dense_ff {
            return Err(bad(format!(
                "has dims {dims:?}; the dense width is {}",
                self.dense_ff
            )));
        }
        let mut out = dims.to_vec();
        out[axis] = DENSE_FF;
        Ok(Some(out))
    }

    fn lines(&self) -> Vec<String> {
        let sites: Vec<String> = self
            .kept()
            .map(|(e, f)| format!("source site {e} at fixture layer {f}"))
            .collect();
        vec![
            format!(
                "engram {} buckets of {} sites, rows={} (source {}): {}",
                self.primes.len(),
                sites.len(),
                self.fx_rows,
                self.src_rows,
                sites.join(", ")
            ),
            format!(
                "dense ff {} -> {DENSE_FF}, routed and shared ff -> {FIXTURE_FF}, window -> {WINDOW}, top_k -> {TOP_K}",
                self.dense_ff
            ),
        ]
    }
}

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds are the source layers'
/// kinds: the same mixer, head count, feed-forward kind and site, every width a key keeps the
/// source's, every width the fixture minifies its own, and every tensor of the source layer's type.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    let mismatch = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    if fx.n_block != spec.layers.len() || fx.n_layer != spec.layers.len() {
        return Err(mismatch(
            "layer count".into(),
            format!(
                "{} blocks and {} layers, not {} and no next-token block",
                fx.n_block,
                fx.n_layer,
                spec.layers.len()
            ),
        ));
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        let (got, want) = (
            (fx.dsa(f), fx.heads(f), fx.dense(f), fx.site(f)),
            (src.dsa(l), src.heads(l), src.dense(l), src.site(l)),
        );
        // A site the map holds is a site of the fixture, and a source site it drops is none.
        if got != want {
            return Err(mismatch(
                what,
                format!("(sparse, heads, dense, site) {got:?}, the source's {want:?}"),
            ));
        }
        if fx.clamp[f] != src.clamp[l] || fx.rope_base_of(f) != src.rope_base_of(l) {
            return Err(mismatch(
                what,
                "the routed clamp or the rope base differs from the source's".into(),
            ));
        }
    }
    let widths = |h: &Hparams| {
        (
            h.n_embd,
            h.n_vocab,
            h.mla,
            h.latent,
            h.hc,
            h.indexer.heads,
            h.indexer.d,
            h.indexer.kpool,
            h.dense_lead,
            h.rms_eps,
            h.rope_base,
            h.rope_base_swa,
        )
    };
    if widths(&fx) != widths(&src) {
        return Err(mismatch(
            "kept widths".into(),
            format!("{:?}, the source's {:?}", widths(&fx), widths(&src)),
        ));
    }
    let minified = (
        fx.dense_ff as u64,
        fx.experts.ff as u64,
        fx.window as u64,
        fx.indexer.top_k as u64,
    );
    if minified != (DENSE_FF, FIXTURE_FF, WINDOW, TOP_K) {
        return Err(mismatch(
            "minified widths".into(),
            format!(
                "feed_forward_length {}, expert_feed_forward_length {}, attention.sliding_window {} \
                 and attention.indexer.top_k {}, not {DENSE_FF}, {FIXTURE_FF}, {WINDOW} and {TOP_K}",
                minified.0, minified.1, minified.2, minified.3
            ),
        ));
    }
    let want = crate::arch::noaux::Experts {
        ff: FIXTURE_FF as usize,
        shared_width: FIXTURE_FF as usize * src.experts.n_shared,
        ..src.experts
    };
    if fx.experts != want {
        return Err(mismatch(
            "experts".into(),
            format!(
                "{:?}, the source's at ff {FIXTURE_FF}: {want:?}",
                fx.experts
            ),
        ));
    }
    let (fe, se) = (
        fx.engram
            .as_ref()
            .ok_or_else(|| meta("ple.layers", "is absent"))?,
        src.engram
            .as_ref()
            .ok_or_else(|| meta("ple.layers", "is absent"))?,
    );
    let sites: Vec<usize> = (0..spec.layers.len()).filter(|&f| fx.site(f)).collect();
    if fe.layers != sites
        || (fe.ngram, fe.heads, fe.key_length) != (se.ngram, se.heads, se.key_length)
        || fe.rows >= se.rows
    {
        return Err(mismatch(
            "engram".into(),
            format!("{fe:?}, the source's {se:?}"),
        ));
    }
    // Every tensor of the fixture is the source layer's, in the source layer's type.
    for (_, t) in fixture.iter_tensors() {
        let (src_name, which) = match crate::fixture::split_layer(&t.name) {
            Some((f, rest)) => {
                let l = *spec.layers.get(f).ok_or_else(|| FixtureError::Tensor {
                    name: t.name.clone(),
                    detail: format!("is in layer {f}, past the map's {}", spec.layers.len()),
                })?;
                (blk(l, rest), format!("the source layer {l}'s"))
            }
            None => (t.name.clone(), "the source's".to_string()),
        };
        let ty = source.find(&src_name).map(|(_, s)| s.ty);
        if ty != Some(t.ty) {
            return Err(FixtureError::Tensor {
                name: t.name.clone(),
                detail: format!("is {}, {which} {src_name} is {ty:?}", t.ty),
            });
        }
    }
    Ok(fx)
}
