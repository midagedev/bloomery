//! The Qwen3.6 (`qwen35moe`) gate fixture's spec ([`spec`]): the facts of a
//! Qwen3.6-35B-A3B file the common generator (`crate::fixture`) cannot know,
//! over the rules this architecture shares with Qwen3.8 (the parent module).
//!
//! The source is lmstudio's Q4_K_M file ([`DEFAULT_MODEL`]): 40 layers of 256
//! routed experts, 8 used, the shared expert joined to the routed stacks on
//! every layer (its three parts are of the stack's types), no NextN tensor or
//! key, no PLE and no draft. The fixture is four layers, one of each kind of
//! layer the file has by the type of its experts:
//!
//! | fixture layer | source layer | kind | experts (gate, up, down) and mixer |
//! |---|---|---|---|
//! | f0 | L5 | GDN | Q4_K, Q4_K, Q4_K; `attn_qkv` Q4_K |
//! | f1 | L11 | attention | Q4_K, Q4_K, Q4_K; `attn_v` Q4_K |
//! | f2 | L0 | GDN | Q4_K, Q4_K, Q6_K; `attn_qkv` Q6_K |
//! | f3 | L39 | attention, the last layer | Q4_K, Q4_K, Q6_K; `attn_v` Q6_K |
//!
//! What it keeps: both expert pairs the file has (a Q4_K or a Q6_K down
//! under a Q4_K gate and up), both types of the mixers, and the recurrent to
//! attending step, all on the card and on the host: every layer is
//! card-eligible (the Qwen3.6 body's card leg runs each of the four layers'
//! three stacks), so the planned budget below leaves half of every layer's
//! experts on the host and none of the layers host-only. What it drops: the
//! interval 4 — the fixture writes 2 ([`super::FIXTURE_INTERVAL`]), which re-derives
//! the same layer kinds from the file; the real tier keeps the source's.
//!
//! The unjoined file (unsloth's UD-Q4_K_M: a Q8_0 shared expert, a Q5_K down)
//! is not a second fixture but a fixture variant (`tools/fixture-variants.tsv`).
//!
//! The card budget is planned ([`card_experts`]): under it the written file's
//! own plan holds 128 of the 256 experts on every layer.

use gguf::Split;

use super::{
    D_MAX, D_MIN, Hparams, Variant, check_kinds_as, const_value, expect_hidden, key_rule,
    shexp_gate,
};
use crate::fixture::{
    Budget, CardBudget, CardExperts, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec,
    KeyRule, Tables, Window, meta,
};
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::workstation::{self, RTX_3090, SCRATCH};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above).
pub const LAYER_MAP: [usize; 4] = [5, 11, 0, 39];

/// The Qwen3.6 model width: the length of the shared expert's gate vector
/// (`ffn_gate_inp_shexp`), which the family's gate value derives from
/// ([`shexp_gate`]) and its plan holds the source to.
const HIDDEN: u32 = 2048;

/// The positions the card budget's plan runs at: the e2e gate's context
/// (`gate_qwen35moe_e2e.rs`, its `CTX`). A plan at another context moves the
/// KV and the ubatch arena, and with them the experts a card holds.
pub const BUDGET_CTX: u64 = 1088;

/// Card bytes past the m = 1 scratch that the Qwen3.6 gates' placed load
/// counts in the card's scratch: its prompt-unit arena (512 rows) and its
/// placed side, which the gpu crate sizes (`Body35::placed_arena_bytes`) and
/// the model crate cannot call [derived: the gate's `placed scratch:` line
/// at the real file's 40 layers, 211,914,632 B, less the 36 layers the
/// fixture does not have — per layer the slot map 256 · 4 B, the card map
/// 257 · 4 B and the route table over 512 · 9 slots, 37,456 B (`route_bytes`
/// of one expert, which is the larger of the two tables), 39,508 B]. The
/// plan's expert rule fills a card by its bytes, so a figure off by a few MiB
/// moves a layer's count by a few experts; it is the figure of the gate's
/// context [`BUDGET_CTX`] and ubatch.
const PLACED_SCRATCH: u64 = 211_914_632 - 36 * 39_508;

/// The target shards are `<STEM>-0000i-of-0000N.gguf`.
pub const STEM: &str = "qwen35moe-fixture";

/// The real file `verify` opens when none is named: the profile's `MODEL`
/// (`tools/ref/models/qwen35moe.sh`).
pub const DEFAULT_MODEL: &str = "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf";

const TARGET_ARCH: &str = "qwen35moe";

/// The Qwen3.6 fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]] the
/// Qwen3.8 fixture draws its scales in (the file's Q4_K and Q6_K at every K
/// it holds, 512 to 4,096, fit it), the source's ff, [`DEFAULT_MODEL`] as the
/// default source, no draft and the planned card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: TARGET_ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Planned(Budget {
            ctx: BUDGET_CTX,
            card_experts,
        }),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        ff: None,
        shard_bytes: DEFAULT_SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Qwen36,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

/// The written file's per-layer card expert counts, planned as the Qwen3.6
/// gates plan a file they open: the qwen3moe program's machine (its m = 1
/// scratch plus the arena, [`workstation::placed_on`]) over the 3090's spec
/// and [`PLACED_SCRATCH`], the routed experts by the expert rule of the
/// body's card leg (`routed_rule` of `Program::Qwen35Body`: a Q4_K or Q5_K gate
/// and up, a Q4_K, Q5_K or Q6_K down) — not the Qwen3.8 program's, whose
/// downs are Q5_1, Q8_0 or IQ4_NL and which would keep every layer of this
/// file on the host.
fn card_experts(split: &Split, budget: Option<u64>) -> Result<CardExperts, FixtureError> {
    super::card_experts_with(
        split,
        budget,
        |inputs| workstation::placed_on(RTX_3090, inputs.hp.n_trunk, SCRATCH + PLACED_SCRATCH),
        |inputs, machine, levers| {
            inputs.plan_rule(
                machine,
                BUDGET_CTX,
                levers,
                routed_rule(Program::Qwen35Body),
            )
        },
    )
}

/// Qwen3.6's rules.
struct Qwen36;

impl Family for Qwen36 {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        key_rule(suffix)
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf, shexp_gate(HIDDEN))
    }

    fn required_keys(&self) -> &'static [&'static str] {
        &["full_attention_interval"]
    }

    fn tables(&self, source: &Split, _spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        expect_hidden(source, HIDDEN)?;
        Ok(Box::new(Gate))
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

/// The family has no table: the plan's one print is the shared expert's gate
/// value, the one number it derives from the model width.
struct Gate;

impl Tables for Gate {
    fn key(&self, key: &str, _suffix: &str, _v: &gguf::Value) -> Result<gguf::Value, FixtureError> {
        Err(meta(key, "is a table key of a family with no tables"))
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
            "shared-expert gate 1/sqrt({HIDDEN}) = {}",
            shexp_gate(HIDDEN)
        )]
    }
}

/// The engine reads `fixture`'s header as `spec`'s layers whose kinds are the
/// source layers' kinds, at the fixture's own interval: [`check_kinds_as`] of
/// the Qwen3.6 variant.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    check_kinds_as(Variant::Qwen35Moe, spec, fixture, source)
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::super::{FIXTURE_INTERVAL, check_kinds as check_kinds_q38, key_rule};
    use super::{HIDDEN, LAYER_MAP, Qwen36, check_kinds, expect_hidden, shexp_gate, spec};
    use crate::arch::qwen35moe::hparams::Kind;
    use crate::arch::qwen35moe::hparams::tests::keys;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::fixture::{CardBudget, Family, KeyRule};
    use crate::placement::accepts::{Program, leg_runs};

    /// The Qwen3.6 keys of a file of `layers` layers at attention interval
    /// `interval`: [`keys`] (a width of 64) without Qwen3.8's hyper-connection,
    /// indexer, PLE and compress-ratio keys.
    fn q36_keys(layers: u32, interval: u32) -> Vec<(&'static str, V)> {
        keys()
            .into_iter()
            .filter(|(k, _)| {
                !k.starts_with("hyper_connection.")
                    && !k.starts_with("attention.indexer.")
                    && !k.starts_with("ple.")
                    && !matches!(
                        *k,
                        "attention.compress_ratios" | "embedding_length_per_layer_input"
                    )
            })
            .map(|(k, v)| match k {
                "block_count" => (k, V::U32(layers)),
                "full_attention_interval" => (k, V::U32(interval)),
                _ => (k, v),
            })
            .collect()
    }

    /// The tensors of a file whose layer `l` is recurrent unless
    /// `(l + 1) % interval == 0`: each layer's mixer marker and its router
    /// (`attn_q` writes the 4 heads' 16 values and the gate beside them).
    fn q36_tensors(layers: usize, interval: usize) -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> =
            ["token_embd.weight", "output_norm.weight", "output.weight"]
                .iter()
                .map(|n| ((*n).to_string(), vec![1]))
                .collect();
        for l in 0..layers {
            t.push(if (l + 1) % interval == 0 {
                (format!("blk.{l}.attn_q.weight"), vec![64, 128])
            } else {
                (format!("blk.{l}.attn_qkv.weight"), vec![1])
            });
            t.push((format!("blk.{l}.ffn_gate_inp.weight"), vec![1]));
        }
        t
    }

    /// `check_kinds` of a fixture of `layers` layers at `interval` against a
    /// 40-layer source at the file's interval 4, the map `map` of `spec()`.
    fn check(
        tag: &str,
        map: &[usize],
        layers: usize,
        interval: usize,
    ) -> Result<Vec<Kind>, String> {
        let mut spec = spec();
        spec.layers = map.to_vec();
        let source = header_shaped(
            &format!("q36fx-{tag}-src"),
            "qwen35moe",
            &q36_keys(40, 4),
            &[],
            &q36_tensors(40, 4),
        );
        let fixture = header_shaped(
            &format!("q36fx-{tag}-fx"),
            "qwen35moe",
            &q36_keys(layers as u32, interval as u32),
            &[],
            &q36_tensors(layers, interval),
        );
        let src = gguf::Split::open(&source).expect("the synthetic header opens");
        let fx = gguf::Split::open(&fixture).expect("the synthetic header opens");
        let checked = check_kinds(&spec, &fx, &src)
            .map(|hp| hp.kinds)
            .map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&fixture);
        checked
    }

    /// The map's four layers read as the kinds they were chosen for — a
    /// recurrent layer, an attending one, a recurrent one and the last layer
    /// attending — at the fixture's interval 2, out of a source at its own 4;
    /// a map whose first layer is an attending one is refused by name, and
    /// so is a fixture written at the source's interval.
    #[test]
    fn the_map_reads_as_the_kinds_it_was_chosen_for() {
        let kinds = check("ok", &LAYER_MAP, 4, FIXTURE_INTERVAL as usize)
            .expect("a fixture of the map's four layers");
        assert_eq!(
            kinds,
            [
                Kind::DeltaRule,
                Kind::Attention,
                Kind::DeltaRule,
                Kind::Attention
            ]
        );
        let err = check("map", &[3, 11, 0, 39], 4, FIXTURE_INTERVAL as usize)
            .expect_err("source layer 3 attends, fixture layer 0 does not");
        assert!(err.contains("layer 0 (source 3)"), "{err}");
        let err = check("interval", &LAYER_MAP, 4, 4).expect_err("the source's interval");
        assert!(err.contains("full_attention_interval"), "{err}");
    }

    /// The Qwen3.8 check refuses a Qwen3.6 pair by its variant, and the
    /// Qwen3.6 one takes it: the variant is the check's argument.
    #[test]
    fn the_variant_is_the_checks_argument() {
        let spec = spec();
        let source = header_shaped(
            "q36fx-var-src",
            "qwen35moe",
            &q36_keys(40, 4),
            &[],
            &q36_tensors(40, 4),
        );
        let fixture = header_shaped(
            "q36fx-var-fx",
            "qwen35moe",
            &q36_keys(4, 2),
            &[],
            &q36_tensors(4, 2),
        );
        let src = gguf::Split::open(&source).expect("the synthetic header opens");
        let fx = gguf::Split::open(&fixture).expect("the synthetic header opens");
        let q38 = check_kinds_q38(&spec, &fx, &src).map(|_| ());
        let q36 = check_kinds(&spec, &fx, &src).map(|_| ());
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&fixture);
        let err = q38.expect_err("a qwen35moe pair is not a qwen4exp one");
        assert!(err.to_string().contains("variant"), "{err}");
        q36.expect("the Qwen3.6 check takes it");
    }

    /// The shared expert's gate vector is `1/√width` rounded once, 0.022_097_087
    /// at Qwen3.6's 2,048; a source of another width is refused by name.
    #[test]
    fn the_shared_expert_gate_is_derived_from_the_width() {
        assert_eq!(shexp_gate(HIDDEN).to_bits(), 0.022_097_087_f32.to_bits());
        assert_eq!(
            Qwen36.const_value("ffn_gate_inp_shexp.weight"),
            Some(shexp_gate(HIDDEN))
        );
        let path = header_shaped(
            "q36fx-hid",
            "qwen35moe",
            &q36_keys(4, 2),
            &[],
            &q36_tensors(4, 2),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let checked = expect_hidden(&split, HIDDEN);
        let _ = std::fs::remove_file(&path);
        let err = checked.expect_err("a source of width 64");
        assert!(err.to_string().contains("is 64, not the 2048"), "{err}");
    }

    /// Every 1-D tensor of the file has a fill rule, and every architecture
    /// key it carries has a key rule (the real file's `qwen35moe.*` keys, by
    /// suffix): a name or a key the rules miss is refused by the plan, on the
    /// box, after the header read.
    #[test]
    fn the_rules_cover_the_files_one_dimensional_tensors_and_keys() {
        let family = Qwen36;
        for (leaf, want) in [
            ("attn_norm.weight", 1.0),
            ("post_attention_norm.weight", 1.0),
            ("output_norm.weight", 1.0),
            ("ssm_norm.weight", 1.0),
            ("attn_q_norm.weight", 1.0),
            ("attn_k_norm.weight", 1.0),
            ("ssm_a", -1.0),
            ("ssm_dt.bias", 0.0),
        ] {
            assert_eq!(family.const_value(leaf), Some(want), "{leaf}");
        }
        assert!(family.spread_value("ffn_gate_inp_shexp.weight").is_none());
        for suffix in [
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.layer_norm_rms_epsilon",
            "attention.value_length",
            "block_count",
            "context_length",
            "embedding_length",
            "expert_count",
            "expert_feed_forward_length",
            "expert_shared_feed_forward_length",
            "expert_used_count",
            "full_attention_interval",
            "rope.dimension_count",
            "rope.dimension_sections",
            "rope.freq_base",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ] {
            assert!(key_rule(suffix).is_some(), "{suffix}");
            assert_eq!(family.key_rule(suffix), key_rule(suffix), "{suffix}");
        }
        assert_eq!(
            family.key_rule("full_attention_interval"),
            Some(KeyRule::Set(FIXTURE_INTERVAL))
        );
    }

    /// The spec's facts: the architecture it declares, the map, no ratios, no
    /// draft and no sidecar, a planned budget at the e2e gate's context; and
    /// the Qwen3.6 body's card leg, whose rule the budget plans under, runs
    /// both expert pairs the map holds (a Q4_K gate and up with a Q4_K and with
    /// a Q6_K down).
    #[test]
    fn the_spec_is_the_map_and_its_card_leg_runs_its_pairs() {
        let spec = spec();
        assert_eq!(spec.arch, "qwen35moe");
        assert_eq!(spec.layers, [5, 11, 0, 39]);
        assert!(spec.ratios.is_empty() && spec.draft.is_none() && spec.sidecar.is_none());
        assert!(
            matches!(spec.card_budget, CardBudget::Planned(b) if b.ctx == 1088),
            "a planned budget at the e2e gate's context"
        );
        for down in [GgmlType::Q4_K, GgmlType::Q6_K] {
            let pair = [GgmlType::Q4_K, GgmlType::Q4_K, down];
            assert_eq!(leg_runs(Program::Qwen35Body, pair), Some(true), "{down}");
        }
    }
}
