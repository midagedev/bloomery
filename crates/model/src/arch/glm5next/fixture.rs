//! The GLM-5.3-Flash (`glm5next`) gate fixture's spec ([`spec`]): the facts of
//! a glm5next file the common generator (`crate::fixture`) cannot know.
//!
//! The fixture is seven blocks — six trunk layers and the NextN layer — and
//! the source's globals. Every layer is a source layer, renamed
//! `blk.L.` → `blk.f.`:
//!
//! | fixture layer | source layer | mixer | feed-forward | why |
//! |---|---|---|---|---|
//! | f0 | L0 | KDA | dense | the first of the dense prefix |
//! | f1 | L1 | KDA | dense | the second: the fixture's dense prefix is two layers, `leading_dense_block_count` 2 |
//! | f2 | L4 | KDA | routed, Q4_K gate·up, Q5_K down (card-eligible) | a routed KDA layer after the prefix |
//! | f3 | L7 | latent | routed, as f2 | the latent mixer with its indexer and pools; index 3 as the real file's first latent layer |
//! | f4 | L8 | KDA | routed, as f2 | the KDA layer after a latent one |
//! | f5 | L11 | latent | routed, Q5_K gate·up, Q6_K down (host-only) | the only source layer with Q5_K gate·up, and a Q6_K down, which `place::card_routed` does not run on the card: the layer keeps all its experts on the host |
//! | f6 | L45 | latent, NextN | routed, as f2 | the NextN layer, which must be the last: the file's `nextn_predict_layers` is believed only while the last layer carries `nextn.eh_proj` |
//!
//! What it keeps: every (role, type) pair of the source's routed stacks (gate
//! and up Q4_K and Q5_K, down Q5_K and Q6_K), both mixers and a latent layer
//! after a KDA one and the reverse, the dense block, the NextN layer's tensors,
//! and the layer kinds' order at the indexes the real file has (f0 to f2 KDA, f3
//! latent, f4 KDA). What it drops: 38 trunk layers and the third dense one.
//!
//! The routed experts' ff is [`FIXTURE_FF`] in place of the source's 2,048,
//! on the three stacks; the shared expert keeps its own
//! (`expert_shared_feed_forward_length`, 2,048: the engine reads it as its own
//! width, `place.rs`'s `shared_ff`).
//!
//! A map that would change a constant the kernels are built for (HC streams 4,
//! KDA head 128, kv_lora 512, indexer 128 × 32 and its pool of 4, 288 experts
//! 8 used, conv 4: [`constants`], the engine's own check in `gpu-glm5next`
//! `body.rs` `dims_of`) is refused by name, in the source and in the written
//! file, as is a map that does not end in the source's NextN layer: the reader
//! ([`Hparams::read`]) takes a file whose last layer carries no
//! `nextn.eh_proj` for a trunk of every layer and says nothing.
//!
//! The card budget ([`Budget`]) is planned, not fixed: under it the plan of the
//! written file with its NextN layer, at the e2e gate's context, holds half
//! of the experts, 144 of 288, on every card-eligible layer (f2 to f4).

use gguf::{Split, Value};

use super::hparams::Hparams;
use super::place::{NextnInputs, PlanInputs};
use crate::fixture::{
    Budget, CardBudget, CardExperts, Family, FixtureError, FixtureSpec, KeyRule, Tables, Window,
    int_like, meta,
};
use crate::placement::{PlanLevers, workstation};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 7] = [0, 1, 4, 7, 8, 11, 45];

/// The routed experts' ff the fixture writes in place of the source's 2,048: a
/// multiple of 256, as the expert stacks' down K-quant blocks need.
pub const FIXTURE_FF: u64 = 512;

/// The d window, [2^-14, 2^-6]: normal f16 values every GLM type's scale
/// fits at every K the fixture holds (K from 4 to 16,384; the Q6_K down at
/// K = 512 needs a narrow band to keep `d` above 2^-14).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The positions the card budget's plan runs at: the e2e and MTP gates'
/// context (`gate_glm5next_e2e.rs`, its `CTX`). A plan at another context
/// moves the KV and the prompt batch's reserve, and with them the experts a
/// card holds.
pub const BUDGET_CTX: u64 = 3136;

/// The most tensor data a target shard holds. The real file is a six-shard
/// split set: a cap of 4 GiB cuts the fixture's about 8.6 GB into shards with
/// a layer across a boundary, so the engine reads a split set here too.
pub const SHARD_BYTES: u64 = 4 << 30;

/// The target shards are `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "glm53-fixture";

/// The real first shard `verify` opens when none is named: the profile's
/// (`tools/ref/models/glm5next.sh`).
pub const DEFAULT_MODEL: &str =
    "/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";

const ARCH: &str = "glm5next";

/// The `ssm_a` constant: the real file's mean over a layer's 64 heads is
/// −5.07 [measured, block 4], and with `ssm_dt.bias` −1.07 the decay
/// `exp(lb · sigmoid(−(f + dt)·sa))` (`crates/gpu/src/linear/conv.rs`,
/// `kda_decay`) sits near 1 for a small `f`, as the real file's does; any
/// finite value keeps it in [e^lb, 1].
const SSM_A: f32 = -5.0;
/// The `ssm_dt.bias` constant, the real file's mean at block 4 (−1.07)
/// [measured].
const SSM_DT_BIAS: f32 = -1.0;

/// The GLM-5.3-Flash fixture: [`LAYER_MAP`], the window
/// [[`D_MIN`], [`D_MAX`]], [`FIXTURE_FF`], the profile's file as the default
/// source, no draft file (the NextN layer is in the file) and the planned
/// card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Planned(Budget {
            ctx: BUDGET_CTX,
            card_experts,
        }),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        ff: Some(FIXTURE_FF),
        shard_bytes: SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Glm,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The coverage table of the architecture keys; `None` is a key the map does
/// not cover. The copy list is every key [`Hparams::read`] reads that is not a
/// function of the map.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 37] = [
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "vocab_size",
        "nextn_predict_layers",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "attention.layer_norm_epsilon",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.key_length",
        "attention.value_length",
        "attention.key_length_mla",
        "attention.value_length_mla",
        "rope.dimension_count",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "kda.head_dim",
        "kda.gate_lower_bound",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
        "attention.indexer.kpool",
        "attention.indexer.index_share_mtp",
        "hyper_connection.count",
        "hyper_connection.sinkhorn_iterations",
        "hyper_connection.epsilon",
        "expert_count",
        "expert_used_count",
        "expert_group_count",
        "expert_group_used_count",
        "expert_gating_func",
        "expert_shared_feed_forward_length",
        "expert_shared_count",
        "expert_weights_scale",
        "expert_weights_norm",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "expert_feed_forward_length" => KeyRule::Ff,
        "leading_dense_block_count" => KeyRule::Table,
        "attention.head_count_kv" | "swiglu_clamp_exp" | "swiglu_clamp_shexp" => KeyRule::PerLayer,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of a 1-D F32 tensor, by its name without the `blk.N.` prefix:
/// the norms' gains 1, the selection and index-norm biases and the
/// hyper-connection bases 0, the hyper-connection scales 1, and the KDA
/// decay's `ssm_a` and `ssm_dt.bias` ([`SSM_A`], [`SSM_DT_BIAS`]).
pub fn const_value(leaf: &str) -> Option<f32> {
    const RULES: [(&str, f32); 18] = [
        ("attn_norm.weight", 1.0),
        ("ffn_norm.weight", 1.0),
        ("attn_q_a_norm.weight", 1.0),
        ("attn_kv_a_norm.weight", 1.0),
        ("ssm_norm.weight", 1.0),
        ("indexer.k_norm.weight", 1.0),
        ("indexer.k_norm.bias", 0.0),
        ("output_norm.weight", 1.0),
        ("nextn.enorm.weight", 1.0),
        ("nextn.hnorm.weight", 1.0),
        ("nextn.shared_head_norm.weight", 1.0),
        ("exp_probs_b.bias", 0.0),
        ("hc_attn_base.weight", 0.0),
        ("hc_ffn_base.weight", 0.0),
        ("hc_attn_scale.weight", 1.0),
        ("hc_ffn_scale.weight", 1.0),
        ("ssm_a", SSM_A),
        ("ssm_dt.bias", SSM_DT_BIAS),
    ];
    RULES.iter().find(|(n, _)| *n == leaf).map(|&(_, v)| v)
}

/// The constants the kernels are built for, as (key, the file's value, the
/// kernels'): `gpu-glm5next`'s `body.rs` `dims_of` refuses a file that differs
/// in any of them.
pub fn constants(hp: &Hparams) -> [(&'static str, usize, usize); 9] {
    [
        ("hyper_connection.count", hp.hc.streams, 4),
        ("kda.head_dim", hp.kda_head_dim, 128),
        ("attention.kv_lora_rank", hp.kv_lora, 512),
        ("attention.indexer.key_length", hp.indexer.head_dim, 128),
        ("attention.indexer.head_count", hp.indexer.n_head, 32),
        ("attention.indexer.kpool", hp.indexer.kpool, 4),
        ("expert_count", hp.n_expert, 288),
        ("expert_used_count", hp.n_used, 8),
        ("ssm.conv_kernel", hp.conv, 4),
    ]
}

/// The first of `hp`'s constants that differs from the kernels', by name.
fn refuse_constants(hp: &Hparams, whose: &str) -> Result<(), FixtureError> {
    match constants(hp).into_iter().find(|(_, got, want)| got != want) {
        None => Ok(()),
        Some((key, got, want)) => Err(meta(
            &format!("{ARCH}.{key}"),
            format!(
                "is {got} in {whose}; the kernels are built for {want} (gpu-glm5next body.rs \
                 dims_of)"
            ),
        )),
    }
}

/// The written file's per-layer card expert counts, planned through the
/// placement planner on the gate card ([`workstation::plan_gate`], the 3090) at
/// [`BUDGET_CTX`] positions under `budget` bytes, with its NextN layer loaded
/// (two KDA lanes, the layer's card bytes and arena reserved out of the
/// budget) — the plan the e2e, MTP and serve gates make of a file they open.
/// `describe`, not `read`: the gates' plan of a file needs no feature check to
/// count experts.
fn card_experts(split: &Split, budget: Option<u64>) -> Result<CardExperts, FixtureError> {
    let inputs = PlanInputs::describe(split)?;
    let nextn = NextnInputs::read(&inputs)?;
    let machine = workstation::plan_gate(inputs.model.layers);
    let levers = PlanLevers {
        card_budget_bytes: budget,
    };
    let plan = inputs
        .plan_nextn(&machine, BUDGET_CTX, &levers, &nextn)
        .map_err(|e| FixtureError::Budget(e.to_string()))?;
    Ok(CardExperts {
        per_layer: plan.plan.n_l.clone(),
        experts: inputs.model.experts,
    })
}

/// glm5next's rules.
struct Glm;

impl Family for Glm {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn required_keys(&self) -> &'static [&'static str] {
        &["leading_dense_block_count", "nextn_predict_layers"]
    }

    /// The routed stacks only: gate and up carry the ff in their second dim,
    /// down in its first. The shared expert has its own width.
    fn ff_axis(&self, leaf: &str) -> Option<usize> {
        match leaf {
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" => Some(1),
            "ffn_down_exps.weight" => Some(0),
            _ => None,
        }
    }

    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        Ok(Box::new(MapTables::read(source, &spec.layers)?))
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

/// What the map decides beyond renaming: the dense prefix's length, which the
/// file's `leading_dense_block_count` states.
struct MapTables {
    dense: usize,
    trunk: usize,
    nextn: usize,
}

impl MapTables {
    /// The map `layers` read against `source`: the kernels' constants kept,
    /// the NextN layer the map's last and its only one past the trunk, the
    /// dense layers a prefix of the map.
    fn read(source: &Split, layers: &[usize]) -> Result<MapTables, FixtureError> {
        let hp = Hparams::read(source)?;
        refuse_constants(&hp, "the source")?;
        let key = format!("{ARCH}.nextn_predict_layers");
        let n_nextn = hp.n_layer - hp.n_trunk;
        if n_nextn != 1 {
            return Err(meta(
                &key,
                format!(
                    "reads {n_nextn} next-token layers; the fixture is built for the one the \
                     NextN load runs (`place::NextnInputs`)"
                ),
            ));
        }
        let nextn = hp.n_trunk;
        let Some((last, trunk)) = layers.split_last() else {
            return Err(meta(&key, "the map holds no layer"));
        };
        if *last != nextn {
            return Err(meta(
                &key,
                format!(
                    "the map {layers:?} does not end in the source's NextN layer {nextn}: the \
                     written file would say {n_nextn} and its last layer carry no \
                     `nextn.eh_proj`, which the reader takes for a trunk of every layer"
                ),
            ));
        }
        if let Some(&l) = trunk.iter().find(|&&l| l >= hp.n_trunk) {
            return Err(meta(
                &key,
                format!(
                    "the map holds layer {l}, past the source's trunk of {}",
                    hp.n_trunk
                ),
            ));
        }
        let dense_at = |l: usize| l < hp.dense_lead;
        let dense = trunk.iter().take_while(|&&l| dense_at(l)).count();
        if let Some(&l) = trunk[dense..].iter().find(|&&l| dense_at(l)) {
            return Err(meta(
                &format!("{ARCH}.leading_dense_block_count"),
                format!(
                    "the map {layers:?} holds the dense layer {l} after a routed one; the key \
                     counts a prefix"
                ),
            ));
        }
        Ok(MapTables {
            dense,
            trunk: trunk.len(),
            nextn,
        })
    }
}

impl Tables for MapTables {
    fn key(&self, key: &str, suffix: &str, v: &Value) -> Result<Value, FixtureError> {
        match suffix {
            "leading_dense_block_count" => int_like(key, v, self.dense as u64),
            _ => Err(meta(
                key,
                "is a table key the glm5next map does not compute",
            )),
        }
    }

    fn dims(
        &self,
        _name: &str,
        _layer: Option<usize>,
        _leaf: &str,
        _dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        Ok(None)
    }

    fn lines(&self) -> Vec<String> {
        vec![format!(
            "glm map: {} dense layers, {} routed, then the NextN layer (source layer {})",
            self.dense,
            self.trunk - self.dense,
            self.nextn
        )]
    }
}

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds are the
/// source layers' kinds, with the kernels' constants kept, the dense layers a
/// prefix and the NextN layer believed.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    let bad = |what: String, detail: String| FixtureError::Mismatch { what, detail };
    refuse_constants(&fx, "the fixture")?;
    if fx.n_layer != spec.layers.len() {
        return Err(bad(
            "layer count".into(),
            format!("{}, not {}", fx.n_layer, spec.layers.len()),
        ));
    }
    if fx.n_layer - fx.n_trunk != 1 {
        return Err(bad(
            "nextn_predict_layers".into(),
            format!(
                "the reader believes {} next-token layers; the last layer's `nextn.eh_proj` is \
                 what it needs",
                fx.n_layer - fx.n_trunk
            ),
        ));
    }
    for (f, &l) in spec.layers.iter().enumerate() {
        let what = format!("layer {f} (source {l})");
        if fx.kinds[f] != src.kinds[l] {
            return Err(bad(
                what,
                format!("kind {:?}, the source's {:?}", fx.kinds[f], src.kinds[l]),
            ));
        }
        if f < fx.n_trunk && (f < fx.dense_lead) != (l < src.dense_lead) {
            return Err(bad(
                what,
                format!(
                    "dense {} with leading_dense_block_count {}, the source's {} with {}",
                    f < fx.dense_lead,
                    fx.dense_lead,
                    l < src.dense_lead,
                    src.dense_lead
                ),
            ));
        }
        if fx.limit_exp[f] != src.limit_exp[l] || fx.limit_shexp[f] != src.limit_shexp[l] {
            return Err(bad(
                what,
                format!(
                    "swiglu clamps {} and {}, the source's {} and {}",
                    fx.limit_exp[f], fx.limit_shexp[f], src.limit_exp[l], src.limit_shexp[l]
                ),
            ));
        }
    }
    let want_ff = spec.ff.map_or(src.expert_ff, |f| f as usize);
    for (key, a, b) in [
        ("n_embd", fx.n_embd, src.n_embd),
        ("n_head", fx.n_head, src.n_head),
        ("n_vocab", fx.n_vocab, src.n_vocab),
        ("n_ctx_train", fx.n_ctx_train, src.n_ctx_train),
        ("q_lora", fx.q_lora, src.q_lora),
        ("head_k", fx.head_k, src.head_k),
        ("head_v", fx.head_v, src.head_v),
        ("n_shared", fx.n_shared, src.n_shared),
        ("shared_ff", fx.shared_ff, src.shared_ff),
        ("dense_ff", fx.dense_ff, src.dense_ff),
        ("expert_ff", fx.expert_ff, want_ff),
    ] {
        if a != b {
            return Err(bad(key.into(), format!("{a}, want {b}")));
        }
    }
    if fx.defaults != src.defaults {
        return Err(bad(
            "the keys read by default".into(),
            format!("{:?}, the source's {:?}", fx.defaults, src.defaults),
        ));
    }
    Ok(fx)
}
