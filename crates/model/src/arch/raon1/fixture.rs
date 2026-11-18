//! The Raon-1-mini-preview (`raon1`) gate fixture's spec ([`spec`]): the facts of a `raon1` file
//! the common generator (`crate::fixture`) cannot know. The trunk mixes Raon delta-attention (RDA)
//! layers, where `attention.head_count_kv` is 0, and gated latent-attention layers, where it is 1;
//! every layer joins its sublayers through four hyper-connection (mHC) streams, the first layer's
//! block is dense and the rest route 6 of 128 experts with a shared one, in one unsplit file.
//!
//! The fixture is five blocks and the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`; the five are one of each of the file's five layer signatures (a mixer, a
//! block and a type of the routed down stack):
//!
//! | fixture layer | source layer | mixer | block | the file's layers of its signature |
//! |---|---|---|---|---|
//! | f0 | L0 | RDA | dense | L0 alone: `ffn_down` Q6_K, `ffn_gate` and `ffn_up` Q4_K |
//! | f1 | L1 | RDA | routed, `ffn_down_exps` Q8_0 | L1, L2, L5, L8, L14, L17, L20 to L22 |
//! | f2 | L3 | latent | routed, `ffn_down_exps` Q5_0 | L3, L7, L15, L19: the Q8_0 `attn_gate`, `attn_k_b`, `attn_q_b` beside the Q6_K latent projections |
//! | f3 | L4 | RDA | routed, `ffn_down_exps` Q5_0 | L4, L6, L9, L10, L12, L13, L16, L18 |
//! | f4 | L11 | latent | routed, `ffn_down_exps` Q8_0 | L11, L23, L24 |
//!
//! What it keeps: both mixer kinds, the dense and the routed block, every (tensor stem, type) pair
//! of the source's layers, and the real globals (`token_embd` Q4_K and `output` Q6_K at their
//! 259,840 rows). What it drops: 20 layers, each of a signature the five hold. [`check_map`]
//! refuses a map that loses a kind or a pair, by name, at the plan and at `verify`. Five layers,
//! not four, because the DFlash2 draft taps five of the target's layers and its `target_layers`
//! are strictly increasing: the fixture draft reads all five.
//!
//! Nothing in the header moves with the map but `block_count` and `attention.head_count_kv`, taken
//! at the map's layers; `leading_dense_block_count` stays 1 because the map's first layer is the
//! dense one.
//!
//! The draft fixture ([`DRAFT_FILE`]) is the real draft's tensors at their shapes with random
//! Q8_0 weights, `target_layers` moved from the source's [2, 7, 12, 17, 22] to the fixture's
//! [0, 1, 2, 3, 4], and its norms 1. `attn_conv_base` and `ffn_conv_base` are 1 per element: the
//! real tables are tap 0 near 1 and tap 1 near 0 (measured on the repo's draft: means 1.0 to 1.1
//! and −0.15 to 0.02), which a one-value rule cannot hold, so a clause that reads those values
//! stays on the real draft.
//!
//! The fills (the generator's matrix rule is σ 1/√K): the norm gains 1; the delta layers' `ssm_a`
//! is −1.0 and `ssm_dt.bias` −5.2, the medians of the file's 18 delta layers' own medians
//! [measured on the repo's Q4_K_M file: −1.03 and −5.17]; the router's selection bias and the
//! four mHC tables are spreads per element ([`BIAS_HALF_WIDTH`], [`BASE_HALF_WIDTH`],
//! [`SCALE_HALF_WIDTH`]), so a body that broadcasts one value or skips one fails. The real tables
//! are structured (the bases hold −3 to 0.77, the scales −6 to 12, the bias is 11 to 14 with a
//! spread of 0.5); the fixture draws the mechanism, not the values. The card budget is fixed.

use gguf::Split;

use gguf::Value;

use super::hparams::{Hparams, Kind};
use crate::arch::dflash2;
use crate::fixture::hybrid;
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, DraftRules, DraftSpec, Family, FixtureError, FixtureSpec,
    KeyRule, Kvs, NoTables, Tables, Window, int_like, items, meta, unsigned,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 5] = [0, 1, 3, 4, 11];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixture
/// holds (2,560 for the projections and globals, 576 for the latent up-projections, 1,088 for the
/// routed and shared down, 12,032 for the dense down).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The half-width of the router's selection bias (`exp_probs_b.bias`), drawn per expert: wide
/// enough that it steers the pick, narrow enough that the sigmoid scores still decide it.
pub const BIAS_HALF_WIDTH: f32 = 0.05;

/// The half-width of the mHC bases (`hc_attn_base`, `hc_ffn_base`), drawn per element.
pub const BASE_HALF_WIDTH: f32 = 1.0;

/// The half-width of the mHC scales (`hc_attn_scale`, `hc_ffn_scale`), drawn per element.
pub const SCALE_HALF_WIDTH: f32 = 2.0;

/// The card byte budget the fixture records: above the fixture's tensor data (about 4 GB), with
/// room for the cache and the scratch of the longest context a gate runs.
pub const CARD_BUDGET: u64 = 8 << 30;

/// The target file is `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "raon1-fixture";

/// The DFlash2 draft fixture's path inside the fixture directory.
pub const DRAFT_FILE: &str = "draft/raon1-fixture-draft.gguf";

/// The layers the real draft taps, as its header names them.
const SOURCE_TAPS: [u64; 5] = [2, 7, 12, 17, 22];

/// The real file `verify` opens when none is named: the profile's (`tools/ref/models/raon1.sh`).
pub const DEFAULT_MODEL: &str = "/models/Raon-1-mini-preview/raon-1-mini-preview-Q4_K_M.gguf";

const ARCH: &str = "raon1";

/// The kinds the map keeps one of each.
const KEPT: [Kind; 2] = [Kind::Rda, Kind::Mla];

/// The Raon-1-mini-preview fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the
/// profile's file as the default source, no ff override and a fixed card budget.
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
        draft: Some(DraftSpec {
            arch: crate::arch::DFLASH,
            file: DRAFT_FILE,
            default_source: default_draft_source,
            rules: &Dflash2,
        }),
        sidecar: None,
        family: &Raon1,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

fn default_draft_source() -> Option<String> {
    std::env::var("BLOOMERY_DSPARK_MODEL").ok()
}

/// The coverage table of the architecture keys; `None` is a key the map does not cover. The copy
/// list is every key [`Hparams::read`] reads that is not a function of the map; the per-layer key
/// follows the map, and `nextn_predict_layers` may be present and must be 0.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 33] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "feed_forward_length",
        "leading_dense_block_count",
        "attention.head_count",
        "attention.key_length",
        "attention.value_length",
        "attention.key_length_mla",
        "attention.value_length_mla",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.layer_norm_rms_epsilon",
        "attention.latent_norm_rms_epsilon",
        "rope.dimension_count",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
        "kda.head_dim",
        "kda.gate_lower_bound",
        "hyper_connection.count",
        "hyper_connection.epsilon",
        "hyper_connection.magnitude",
        "activation.situ_beta",
        "activation.situ_linear_beta",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_count",
        "expert_shared_feed_forward_length",
        "expert_gating_func",
        "expert_weights_scale",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count_kv" => KeyRule::PerLayer,
        "nextn_predict_layers" => KeyRule::Zero("the reader runs no next-token layer"),
        "expert_weights_norm" | "expert_group_count" | "expert_group_used_count" => KeyRule::Copy,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.`: the norm gains 1, and
/// the delta layers' `ssm_a` and `ssm_dt.bias` (see the module's fills).
pub fn const_value(leaf: &str) -> Option<f32> {
    const GAINS: [&str; 8] = [
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        "output_norm.weight",
        "ssm_norm.weight",
        "attn_q_a_norm.weight",
        "attn_kv_a_norm.weight",
    ];
    match leaf {
        "ssm_a" => Some(-1.0),
        "ssm_dt.bias" => Some(-5.2),
        "attn_conv_base" | "ffn_conv_base" | "enc.output_norm.weight" => Some(1.0),
        _ => GAINS.contains(&leaf).then_some(1.0),
    }
}

/// The spread of an F32 tensor the family names: the router's selection bias and the mHC tables.
pub fn spread_value(leaf: &str) -> Option<f32> {
    match leaf {
        "exp_probs_b.bias" => Some(BIAS_HALF_WIDTH),
        "hc_attn_base.weight" | "hc_ffn_base.weight" => Some(BASE_HALF_WIDTH),
        "hc_attn_scale.weight" | "hc_ffn_scale.weight" => Some(SCALE_HALF_WIDTH),
        _ => None,
    }
}

/// `layers` keeps the source's facts: one layer of each of [`KEPT`] and every (stem, type) pair
/// any source layer holds ([`hybrid::check_map`]).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    hybrid::check_map(&src.kinds, &KEPT, source, layers)
}

/// raon1's rules.
struct Raon1;

impl Family for Raon1 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    /// No dim moves with the map: it is only held to [`check_map`].
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
/// layer count and each layer's kind the source layer's, every other fact the source's own, under
/// a map [`check_map`] accepts. Tensor names, dims and types are the plan's, which `verify` holds
/// the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    let want = Hparams {
        n_layer: spec.layers.len(),
        kinds: spec.layers.iter().map(|&l| src.kinds[l]).collect(),
        ..src
    };
    if fx != want {
        return Err(FixtureError::Mismatch {
            what: "the hyperparameters".into(),
            detail: format!("{fx:?}, the source's at the map {want:?}"),
        });
    }
    Ok(fx)
}

/// The DFlash2 draft's rules.
struct Dflash2;

impl DraftRules for Dflash2 {
    /// The real draft's metadata with `target_layers` moved to the fixture's layers in order: the
    /// source's five taps (the header's own, [`SOURCE_TAPS`]) become the fixture's five layers.
    /// Any other tap list, or one longer than the fixture, is refused by name; a draft whose ff
    /// moves with an override is refused too, because the target's ff is not the draft's.
    fn kvs(
        &self,
        draft: &Split,
        n_source: usize,
        n_fixture: usize,
        ff: Option<(u64, u64)>,
    ) -> Result<Kvs, FixtureError> {
        if ff.is_some() {
            return Err(meta(
                &draft.arch_key("feed_forward_length"),
                "the target's ff override does not move the draft's: a Raon fixture takes none",
            ));
        }
        let key = draft.arch_key("target_layers");
        let mut kvs = Vec::new();
        for (k, v) in draft.iter_kv() {
            if k != key {
                kvs.push((k.to_string(), v.clone()));
                continue;
            }
            let a = items(k, v)?;
            let got = a
                .iter()
                .map(|x| unsigned(k, x))
                .collect::<Result<Vec<_>, _>>()?;
            if got != SOURCE_TAPS || got.iter().any(|&l| l as usize >= n_source) {
                return Err(meta(
                    k,
                    format!(
                        "is {got:?}, not the repo's {SOURCE_TAPS:?} of a {n_source}-layer source"
                    ),
                ));
            }
            if got.len() > n_fixture {
                return Err(meta(
                    k,
                    format!(
                        "names {} layers, more target layers than the fixture's {n_fixture}",
                        got.len()
                    ),
                ));
            }
            let moved = (0..got.len() as u64)
                .map(|f| int_like(k, &a[0], f))
                .collect::<Result<_, _>>()?;
            kvs.push((k.to_string(), Value::Array(moved)));
        }
        if !kvs.iter().any(|(k, _)| *k == key) {
            return Err(meta(&key, "is absent"));
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

/// The draft fixture reads as a DFlash2 draft whose taps are the fixture's layers in order; with
/// both files whole, its tensor inventory holds and its tap rows (`target_hidden`: the target's
/// streams of `embedding_length`) and its vocabulary are the target's.
pub fn check_draft(
    spec: &FixtureSpec,
    draft: &Split,
    target: &Split,
    whole: bool,
) -> Result<dflash2::Hparams, FixtureError> {
    let hp = dflash2::Hparams::read(draft)?;
    let want: Vec<usize> = (0..hp.target_layers.len()).collect();
    if hp.target_layers.len() > spec.layers.len() || hp.target_layers != want {
        return Err(FixtureError::Mismatch {
            what: "draft target_layers".into(),
            detail: format!(
                "{:?}, not the fixture's first {} layers {want:?}",
                hp.target_layers,
                want.len()
            ),
        });
    }
    if whole {
        dflash2::check(draft, &hp)?;
        let tp = Hparams::read(target)?;
        let rows = tp.hc.streams * tp.n_embd;
        if hp.target_hidden != rows || hp.n_vocab != tp.n_vocab {
            return Err(FixtureError::Mismatch {
                what: "the draft against its target".into(),
                detail: format!(
                    "target_hidden {} and vocabulary {}, the target's {rows} and {}",
                    hp.target_hidden, hp.n_vocab, tp.n_vocab
                ),
            });
        }
    }
    Ok(hp)
}
