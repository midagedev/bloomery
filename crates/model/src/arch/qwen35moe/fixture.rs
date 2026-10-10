//! The Qwen3.8 (`qwen4exp`) gate fixture's spec ([`spec`]): the facts of a
//! Qwen3.8 file the common generator (`crate::fixture`) cannot know.
//!
//! The fixture is four layers — one GDN without PLE, two QSA, and the GDN
//! PLE site — and the source's globals:
//!
//! | fixture layer | source layer | kind | experts |
//! |---|---|---|---|
//! | f0 | L2 | GDN | Q5_K gate and up, Q8_0 down |
//! | f1 | L3 | QSA | Q4_K gate and up, Q5_1 down |
//! | f2 | L1 | GDN + PLE (the source's `ple.layers` `[1]`, written `[2]`) | as f1 |
//! | f3 | L47 | QSA, the last layer | Q4_K gate and up, Q8_0 down |
//!
//! What it keeps: all four expert kernel pairs, on the card and on the host
//! (q5_K and q8_0 on f0, q4_K and q5_1 on f1 and f2, q4_K and q8_0 on f3),
//! the QSA→GDN step, and PLE on a GDN layer. Every one of the four layers is
//! card-eligible (`place::card_routed` loads each stack type), so the planned
//! budget below leaves half of every layer's experts on the host and none of
//! the layers host-only. What it drops: the interval 4 — the fixture writes 2
//! ([`FIXTURE_INTERVAL`]), which re-derives the same layer kinds from the
//! file (`hparams`, every `(l + 1) % interval`-th layer attending); the real
//! tier keeps the source's.
//!
//! The PLE table (`per_layer_token_embd.weight`, IQ4_NL, 320M rows) is cut
//! to at most [`PLE_TABLE_BYTES`]: its head vocab sizes scaled down and its
//! head offsets rewritten as their prefix sums, so the engine's reader sees
//! the same site geometry over a few million rows instead of 320 million
//! ([`Ple`]).
//!
//! The MTP companion ([`DRAFT_FILE`], beside the target) is the real shared draft's tensors at
//! their shapes with random weights by the same rules, its one layer moved
//! to the fixture's own MTP index and its `block_count` and ratio array
//! rewritten ([`QwenMtp`]).
//!
//! The card budget ([`Budget`]) is planned, not fixed: under it the written
//! file's own plan holds 256 of the 512 experts on every card-eligible
//! layer, half the file's experts on the host beside them.

use gguf::{Split, Value};

use super::hparams::{Hparams, Variant};
use super::place;
use super::spec as qspec;
use crate::fixture::{
    Budget, CardBudget, CardExperts, DEFAULT_SHARD_BYTES, DraftRules, DraftSpec, Family,
    FixtureError, FixtureSpec, KeyRule, Kvs, Tables, Window, int_like, items, meta, unsigned,
};
use crate::placement::PlanLevers;
use crate::placement::workstation::RTX_3090;

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [2, 3, 1, 47];

/// The compress ratios the map must read from the source ([`Kind`]): a GDN
/// layer 0, a QSA layer its pool (4 divides `attention.indexer.top_k`
/// 2048). The check that the map still picks the kinds it was chosen for.
pub const FIXTURE_RATIOS: [u64; 4] = [0, 4, 0, 4];

/// The interval the fixture writes in place of the source's 4: with it the
/// four layers' kinds re-derive as [GDN, QSA, GDN, QSA], the source layers'
/// own (`hparams` reads the interval from the file).
pub const FIXTURE_INTERVAL: u64 = 2;

/// The d window, [2^-14, 2^-6]: normal f16 values wide enough for every Qwen3.8 type's scale at every K the file holds — the
/// Q5_1 down's K = 640 needs `d = 2^-7.87`, which V4.1's narrower window
/// refuses.
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The PLE table's cut: the fixture's head vocab sizes are scaled to keep
/// the table at about this many bytes (the source's table is 28.8 GB).
pub const PLE_TABLE_BYTES: u64 = 112 << 20;

/// The shared expert's gate input, `1/√2560` (the model width).
const SHEXP_GATE: f32 = 0.019_764_235;

/// The positions the card budget's plan runs at: the e2e gate's context
/// (`gate_qwen4exp_e2e.rs`, its `CTX`). A plan at another context moves the
/// KV and the ubatch arena, and with them the experts a card holds.
pub const BUDGET_CTX: u64 = 3072;

/// The target shards are `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "qwen38-fixture";
/// The MTP companion's path inside the fixture directory: the real shared
/// draft's file name, beside the target, where the engine looks for a draft
/// when `BLOOMERY_MTP_DRAFT` is unset (`refset`'s `qwen4exp::mtp::draft_file`);
/// under any other name or place a run takes the real draft, which is not the
/// fixture's.
pub const DRAFT_FILE: &str = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

/// The real first shard `verify` opens when none is named: the profile's
/// `QWEN38_FILE` (`tools/ref/models/qwen4exp.sh`).
pub const DEFAULT_MODEL: &str =
    "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
/// The real shared MTP draft `verify` opens when none is named — shared, so
/// the draft fixture borrows the target's `token_embd` and `output` as this
/// load does.
pub const DEFAULT_MTP_MODEL: &str =
    "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

const TARGET_ARCH: &str = "qwen4exp";

/// The Qwen3.8 fixture: [`LAYER_MAP`], [`FIXTURE_RATIOS`], the window
/// [[`D_MIN`], [`D_MAX`]], the source's ff, the Qwen3.8 file as the default
/// source, the MTP companion, and the planned card budget.
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
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        ff: None,
        shard_bytes: DEFAULT_SHARD_BYTES,
        default_source,
        draft: Some(DraftSpec {
            arch: TARGET_ARCH,
            file: DRAFT_FILE,
            default_source: default_mtp_source,
            rules: &QwenMtp,
        }),
        sidecar: None,
        family: &Qwen38,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

fn default_mtp_source() -> Option<String> {
    Some(DEFAULT_MTP_MODEL.to_string())
}

/// The coverage table of the target's architecture keys; `None` is a key
/// the map does not cover. The copy list is every key the qwen4exp readers
/// read (`hparams`, `mtp`) that is not a function of the map.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 38] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.sliding_window",
        "rope.dimension_count",
        "rope.dimension_sections",
        "rope.freq_base",
        "attention.layer_norm_rms_epsilon",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_feed_forward_length",
        "expert_gating_func",
        "expert_weights_norm",
        "expert_weights_scale",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
        "ssm.inner_size",
        "hyper_connection.count",
        "hyper_connection.low_rank",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
        "ple.ngram_size",
        "ple.heads_per_ngram",
        "ple.conv_kernel",
        "ple.eos_token_id",
        "ple.image_token_id",
        "embedding_length_per_layer_input",
        "ple.layer_multipliers",
        "nextn_predict_layers",
        "nextn_shared_target_tensors",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "full_attention_interval" => KeyRule::Set(FIXTURE_INTERVAL),
        "attention.compress_ratios" => KeyRule::Ratios,
        "attention.recurrent_layers" => KeyRule::PerLayer,
        "ple.layers" => KeyRule::LayerIds,
        "ple.head_offsets" | "ple.head_vocab_sizes" => KeyRule::Table,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of a 1-D F32 tensor, by its name without the `blk.N.` prefix:
/// the norms 1 (the mixers', the indexers', the hyper-connection modules',
/// the PLE site's, the head's, the MTP input's), the delta time step's bias
/// 0, `ssm_a` −1, and the shared expert's gate input `1/√2560`.
///
/// `ssm_a` is `−e^A_log` as the converter folds it, so a decay
/// `exp(softplus(a + dt)·ssm_a)` below 1 needs a negative value; −1 keeps the
/// recurrent state bounded however long the run. The gate input's width is
/// the model's `HIDDEN`, so its logit over unit-RMS hidden states is about
/// unit scale (a gain of 1 would saturate the sigmoid at about ±50).
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 17] = [
        ("ssm_norm.weight", 1.0),
        ("attn_q_norm.weight", 1.0),
        ("attn_k_norm.weight", 1.0),
        ("indexer.q_norm.weight", 1.0),
        ("indexer.k_norm.weight", 1.0),
        ("hc_attn_norm.weight", 1.0),
        ("hc_ffn_norm.weight", 1.0),
        ("ple_norm_key.weight", 1.0),
        ("ple_norm_query.weight", 1.0),
        ("ple_norm_conv.weight", 1.0),
        ("output_hc_norm.weight", 1.0),
        ("ssm_a", -1.0),
        ("ssm_dt.bias", 0.0),
        ("ffn_gate_inp_shexp.weight", SHEXP_GATE),
        ("nextn.enorm.weight", 1.0),
        ("nextn.hnorm.weight", 1.0),
        ("nextn.hc_head_norm.weight", 1.0),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The written file's per-layer card expert counts, planned through the
/// placement planner on the 3090's spec (`machine_for_experts` with the
/// ubatch a [`BUDGET_CTX`]-position load runs) under `budget` bytes — the
/// plan the qwen4exp gates make of a file they open at that context.
/// `describe`, not `read`: the gates' own choice, so a chat-surface item
/// `read` refuses plans here as it does there.
fn card_experts(split: &Split, budget: Option<u64>) -> Result<CardExperts, FixtureError> {
    let inputs = place::PlanInputs::describe(split)?;
    let machine = place::machine_for_experts(
        RTX_3090,
        inputs.spec.layers.len(),
        place::UBATCH_PLANNED.min(BUDGET_CTX),
        place::Experts::Card,
    );
    let levers = PlanLevers {
        card_budget_bytes: budget,
    };
    let plan = inputs
        .plan_with(&machine, BUDGET_CTX, &levers, place::Experts::Card)
        .map_err(|e| FixtureError::Budget(e.to_string()))?;
    Ok(CardExperts {
        per_layer: plan.n_l,
        experts: inputs.model.experts,
    })
}

/// Qwen3.8's rules.
struct Qwen38;

impl Family for Qwen38 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn required_keys(&self) -> &'static [&'static str] {
        &["full_attention_interval"]
    }

    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        Ok(Box::new(Ple::read(source, &spec.layers)?))
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

/// The PLE table of one plan: the source's site geometry, and its head
/// vocab sizes scaled down to the cut.
struct Ple {
    /// The site's heads, `ple.heads_per_ngram` per n-gram size but one.
    heads: usize,
    /// The table's value rows, `embedding_length_per_layer_input`.
    row: u64,
    /// The source's rows: the last head range's end.
    rows: u64,
    fx_offsets: Vec<u64>,
    fx_vocabs: Vec<u64>,
    fx_rows: u64,
    offset_tag: Value,
    vocab_tag: Value,
}

/// The PLE table's name, the one tensor the fixture's own dims cut.
const TABLE: &str = "per_layer_token_embd.weight";

impl Ple {
    /// The PLE site of `source`, which must be one of the map `layers`.
    fn read(source: &Split, layers: &[usize]) -> Result<Ple, FixtureError> {
        let key = |s: &str| format!("{TARGET_ARCH}.ple.{s}");
        let get = |s: &str| {
            source
                .value(&key(s))
                .ok_or_else(|| meta(&key(s), "is absent"))
        };
        let ple_layers = get("layers")?;
        let site = match items(&key("layers"), ple_layers)? {
            [v] => unsigned(&key("layers"), v)? as usize,
            [] => {
                return Err(meta(
                    &key("layers"),
                    "is empty: ik leaves PLE inert on a file without the site, and the fixture is \
                     built for it",
                ));
            }
            _ => {
                return Err(meta(
                    &key("layers"),
                    "lists more than one layer; one PLE site is supported",
                ));
            }
        };
        if !layers.contains(&site) {
            return Err(meta(
                &key("layers"),
                format!("names layer {site}, which the map {layers:?} does not hold"),
            ));
        }
        let ngram = unsigned(&key("ngram_size"), get("ngram_size")?)? as usize;
        let per = unsigned(&key("heads_per_ngram"), get("heads_per_ngram")?)? as usize;
        let heads = ngram
            .checked_sub(1)
            .map(|n| n * per)
            .filter(|&h| h > 0)
            .ok_or_else(|| meta(&key("ngram_size"), "leaves no head"))?;
        let o = items(&key("head_offsets"), get("head_offsets")?)?;
        let v = items(&key("head_vocab_sizes"), get("head_vocab_sizes")?)?;
        if o.len() != heads || v.len() != heads {
            return Err(meta(
                &key("head_offsets"),
                format!(
                    "{} offsets and {} vocab sizes for {heads} heads",
                    o.len(),
                    v.len()
                ),
            ));
        }
        let offsets = o
            .iter()
            .map(|x| unsigned(&key("head_offsets"), x))
            .collect::<Result<Vec<_>, _>>()?;
        let vocabs = v
            .iter()
            .map(|x| unsigned(&key("head_vocab_sizes"), x))
            .collect::<Result<Vec<_>, _>>()?;
        let mut rows = 0u64;
        for (h, (&off, &n)) in offsets.iter().zip(&vocabs).enumerate() {
            if off
                .checked_add(n)
                .filter(|&e| n > 0 && e <= i32::MAX as u64)
                .is_none()
            {
                return Err(meta(
                    &key("head_vocab_sizes"),
                    format!("is {n} at head {h}, offset {off}: the range is empty or past an i32"),
                ));
            }
            rows = rows.max(off + n);
        }
        let (_, t) = source.find(TABLE).ok_or_else(|| FixtureError::Tensor {
            name: TABLE.to_string(),
            detail: "is not in the source file".into(),
        })?;
        if t.dims.len() != 2 || t.dims[1] < rows {
            return Err(FixtureError::Tensor {
                name: TABLE.to_string(),
                detail: format!(
                    "has dims {:?}; the PLE heads read rows of {} values, {rows} of them",
                    t.dims,
                    t.dims.first().unwrap_or(&0)
                ),
            });
        }
        // The cut: every head's vocab scaled by the same factor, at least one
        // row each, the offsets their prefix sums.
        let row = t.dims[0];
        let bytes_per_row = t.nbytes / t.dims[1];
        let target_rows = (PLE_TABLE_BYTES / bytes_per_row.max(1)).max(heads as u64);
        let fx_vocabs: Vec<u64> = vocabs
            .iter()
            .map(|&n| (n * target_rows / rows).max(1))
            .collect();
        let mut fx_offsets = Vec::with_capacity(heads);
        let mut acc = 0u64;
        for &n in &fx_vocabs {
            fx_offsets.push(acc);
            acc += n;
        }
        Ok(Ple {
            heads,
            row,
            rows,
            fx_offsets,
            fx_vocabs,
            fx_rows: acc,
            offset_tag: o[0].clone(),
            vocab_tag: v[0].clone(),
        })
    }
}

impl Tables for Ple {
    fn key(&self, key: &str, suffix: &str, _v: &Value) -> Result<Value, FixtureError> {
        match suffix {
            "ple.head_vocab_sizes" => self
                .fx_vocabs
                .iter()
                .map(|&n| int_like(key, &self.vocab_tag, n))
                .collect::<Result<_, _>>()
                .map(Value::Array),
            "ple.head_offsets" => self
                .fx_offsets
                .iter()
                .map(|&o| int_like(key, &self.offset_tag, o))
                .collect::<Result<_, _>>()
                .map(Value::Array),
            _ => Err(meta(key, "is a table key the PLE tables do not compute")),
        }
    }

    fn dims(
        &self,
        _name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        if layer.is_some() || leaf != TABLE {
            return Ok(None);
        }
        if dims.len() != 2 || dims[0] != self.row {
            return Err(FixtureError::Tensor {
                name: TABLE.to_string(),
                detail: format!("has dims {dims:?}; the site reads rows of {}", self.row),
            });
        }
        Ok(Some(vec![self.row, self.fx_rows]))
    }

    fn lines(&self) -> Vec<String> {
        vec![format!(
            "ple table heads={} rows={} (source {}): the head vocab sizes cut to a table of at \
             most {} bytes",
            self.heads, self.fx_rows, self.rows, PLE_TABLE_BYTES,
        )]
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

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds are
/// the source layers' kinds, at the fixture's own interval.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    if fx.variant != src.variant || fx.variant != Variant::Qwen4Exp {
        return Err(bad(
            "variant".into(),
            format!("{:?}, the source's {:?}", fx.variant, src.variant),
        ));
    }
    if fx.n_trunk != spec.layers.len() {
        return Err(bad(
            "layer count".into(),
            format!("{}, not {}", fx.n_trunk, spec.layers.len()),
        ));
    }
    if fx.interval as u64 != FIXTURE_INTERVAL {
        return Err(bad(
            "full_attention_interval".into(),
            format!("is {}, not {}", fx.interval, FIXTURE_INTERVAL),
        ));
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        if fx.kinds[f] != src.kinds[l] || fx.ffns[f] != src.ffns[l] {
            return Err(bad(
                what,
                format!(
                    "kind {:?} ffns {:?}, the source's {:?} {:?}",
                    fx.kinds[f], fx.ffns[f], src.kinds[l], src.ffns[l]
                ),
            ));
        }
    }
    for (key, a, b) in [
        ("n_head", fx.n_head, src.n_head),
        ("n_head_kv", fx.n_head_kv, src.n_head_kv),
        ("head_dim", fx.head_dim, src.head_dim),
        ("n_vocab", fx.n_vocab, src.n_vocab),
        ("n_expert", fx.n_expert, src.n_expert),
        ("n_used", fx.n_used, src.n_used),
        ("expert_ff", fx.expert_ff, src.expert_ff),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, the source's {b}")));
        }
    }
    let (fxe, srce) = (fx.exp.as_ref(), src.exp.as_ref());
    match (fxe, srce) {
        (Some(fxe), Some(srce)) => {
            for (f, &l) in spec.layers.iter().enumerate() {
                if fxe.ratios[f] != srce.ratios[l] {
                    return Err(bad(
                        "attention.compress_ratios".into(),
                        format!(
                            "reads {} at layer {f}, the source layer {l}'s {}",
                            fxe.ratios[f], srce.ratios[l]
                        ),
                    ));
                }
            }
            match (fxe.ple, srce.ple) {
                (Some(f), Some(s)) => {
                    let want = fixture_layer(&spec.layers, s.layer, "the PLE site")?;
                    if f.layer != want
                        || f.ngram != s.ngram
                        || f.heads_per_ngram != s.heads_per_ngram
                        || f.conv != s.conv
                        || f.row != s.row
                    {
                        return Err(bad(
                            "the PLE site".into(),
                            format!(
                                "reads {f:?} where the map moves the source's site {s:?} to layer \
                                 {want}"
                            ),
                        ));
                    }
                }
                (None, None) => {}
                (f, s) => {
                    return Err(bad(
                        "the PLE site".into(),
                        format!("reads {f:?}, the source's {s:?}"),
                    ));
                }
            }
        }
        (None, None) => {}
        (f, s) => {
            return Err(bad(
                "the qwen4exp keys".into(),
                format!("reads {f:?}, the source's {s:?}"),
            ));
        }
    }
    Ok(fx)
}

/// The MTP draft's rules: the layer map [`LAYER_MAP`] the target's spec cuts
/// by, so the draft fixture's metadata and tensor names follow the same cut.
/// A variant spec of another map of the same length writes the draft's
/// decorative per-layer entries by this map, not its own: `kvs` is not given
/// the spec.
pub struct QwenMtp;

impl QwenMtp {
    fn map(&self) -> &'static [usize] {
        &LAYER_MAP
    }
}

impl DraftRules for QwenMtp {
    /// The real draft's metadata with `block_count` moved to the fixture's
    /// own MTP index and `attention.compress_ratios` to the mapped layers'
    /// pools plus the MTP layer's own.
    fn kvs(
        &self,
        draft: &Split,
        n_source: usize,
        n_fixture: usize,
        _ff: Option<(u64, u64)>,
    ) -> Result<Kvs, FixtureError> {
        let map = self.map();
        if n_fixture != map.len() || map.iter().any(|&l| l >= n_source) {
            return Err(FixtureError::Mismatch {
                what: "the MTP rules".into(),
                detail: format!(
                    "are built for the map {map:?} of a {n_source}-layer source, not \
                     {n_fixture} layers"
                ),
            });
        }
        let block_key = draft.arch_key("block_count");
        let ratios_key = draft.arch_key("attention.compress_ratios");
        let mut kvs = Vec::new();
        for (k, v) in draft.iter_kv() {
            if k != block_key && k != ratios_key {
                kvs.push((k.to_string(), v.clone()));
                continue;
            }
            if k == block_key {
                let got = unsigned(k, v)?;
                if got != n_source as u64 + 1 {
                    return Err(meta(
                        k,
                        format!(
                            "is {got}, not the target's {n_source} layers and the next-token one"
                        ),
                    ));
                }
                kvs.push((k.to_string(), int_like(k, v, n_fixture as u64 + 1)?));
                continue;
            }
            let a = items(k, v)?;
            let own = match a.len() {
                n if n == n_source + 1 => a[n_source].clone(),
                n if n == n_source => {
                    return Err(meta(
                        k,
                        format!(
                            "has {n} values, the main layers': the MTP layer would inherit the \
                             last one's pool, and a selecting MTP layer is not built"
                        ),
                    ));
                }
                n => {
                    return Err(meta(
                        k,
                        format!("has {n} values for the draft's {} layers", n_source + 1),
                    ));
                }
            };
            if unsigned(k, &own)? != 0 {
                return Err(meta(
                    k,
                    format!(
                        "gives the MTP layer a pool of {}: a selecting MTP layer is not built",
                        unsigned(k, &own)?
                    ),
                ));
            }
            let tag = &a[0];
            let mapped = map
                .iter()
                .chain(std::iter::once(&n_source))
                .map(|&l| int_like(k, tag, unsigned(k, &a[l])?))
                .collect::<Result<Vec<_>, _>>()?;
            kvs.push((k.to_string(), Value::Array(mapped)));
        }
        if !kvs.iter().any(|(k, _)| *k == block_key) {
            return Err(meta(&block_key, "is absent"));
        }
        Ok(kvs)
    }

    /// `blk.{n_source}.` — the MTP layer of the real draft — renamed to
    /// `blk.{n_fixture}.`; any other layer's tensor is refused by name.
    fn tensor(
        &self,
        name: &str,
        n_source: usize,
        n_fixture: usize,
    ) -> Result<String, FixtureError> {
        let from = format!("blk.{n_source}.");
        if let Some(rest) = name.strip_prefix(&from) {
            return Ok(format!("blk.{n_fixture}.{rest}"));
        }
        if name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(l, _)| l.parse::<usize>().is_ok())
        {
            return Err(FixtureError::Tensor {
                name: name.to_string(),
                detail: format!("is a tensor of another layer; the MTP layer is {n_source}"),
            });
        }
        Ok(name.to_string())
    }

    /// The draft fixture reads, through the engine's own MTP reader, as a
    /// draft of `target`.
    fn check(
        &self,
        _spec: &FixtureSpec,
        draft: &Split,
        target: &Split,
        _whole: bool,
    ) -> Result<(), FixtureError> {
        let read = qspec::read(target).map_err(|e| FixtureError::Mismatch {
            what: "the target's description".into(),
            detail: e.to_string(),
        })?;
        super::mtp::mtp_of(draft, target, &read.spec)
            .map(|_| ())
            .map_err(|e| FixtureError::Mismatch {
                what: "the MTP draft".into(),
                detail: e.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{FIXTURE_INTERVAL, check_kinds, spec};
    use crate::arch::qwen35moe::hparams::tests::{keys, tensors};
    use crate::arch::synthetic::{V, header_shaped};

    /// The map's layers are the fixture's trunk: a fixture that declares a
    /// next-token layer in its four blocks would read its last mapped layer
    /// as unused, and is refused by the layer count.
    #[test]
    fn a_fixture_whose_trunk_is_not_the_map_is_refused() {
        let mut spec = spec();
        spec.layers = vec![0, 1, 2, 3];
        let fixture_keys = |nextn: Option<u32>| {
            let mut kv: Vec<(&str, V)> = keys()
                .into_iter()
                .filter(|(k, _)| *k != "full_attention_interval")
                .collect();
            kv.push(("full_attention_interval", V::U32(FIXTURE_INTERVAL as u32)));
            kv.push(("attention.recurrent_layers", V::I32s(vec![1, 1, 1, 0])));
            kv.extend(nextn.map(|n| ("nextn_predict_layers", V::U32(n))));
            kv
        };
        let check = |tag: &str, nextn: Option<u32>| {
            let source = header_shaped("qfx-src", "qwen4exp", &keys(), &[], &tensors());
            let mut t = tensors();
            if nextn.is_some() {
                t.push(("blk.3.nextn.eh_proj.weight".to_string(), vec![1]));
            }
            let fixture = header_shaped(tag, "qwen4exp", &fixture_keys(nextn), &[], &t);
            let src = gguf::Split::open(&source).expect("the synthetic header opens");
            let fx = gguf::Split::open(&fixture).expect("the synthetic header opens");
            let checked = check_kinds(&spec, &fx, &src);
            let _ = std::fs::remove_file(&source);
            let _ = std::fs::remove_file(&fixture);
            checked
        };
        check("qfx-ok", None).expect("a fixture of the map's four layers");
        let err = check("qfx-nextn", Some(1)).expect_err("its trunk is three layers");
        assert!(err.to_string().contains("layer count: 3, not 4"), "{err}");
    }
}
