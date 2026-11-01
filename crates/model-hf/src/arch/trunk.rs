//! What the Qwen maps read the same way: a text model of grouped-query attention with per-head
//! query and key norms over a feed-forward block, routed or dense ([`Ffn`]). Its `config.json`
//! keys and counts ([`KEYS`], [`COUNTS`], [`Ffn::keys`], [`Ffn::counts`], [`check_counts`]), the
//! header keys they make ([`metadata`]), and the program's tensor and role of each checkpoint
//! tensor that every map names alike: the attention block ([`attention`]), the block norms
//! ([`block_norms`]), the router ([`router`]), a layer's stacked routed experts
//! ([`stacked_experts`]), a layer's dense block ([`dense_ffn`]) and the model-level tensors
//! ([`global`]), and the check that a tensor has values ([`sized`]). A map keeps what its
//! architecture adds (the gated delta rule, the shared expert, the hyper-connections) and the
//! steps that move a tensor into the program's layout.

use gguf::Value;
use hf::modelopt::Plain;
use hf::schema::{Fields, Rule, Rule::*, Type};
use model::arch::names as common;

use super::section::Section;

/// `tie_word_embeddings` of a map whose model has a head of its own: absent or false. A map
/// that must read a folder whose head is the embedding takes [`HEAD_EITHER`] instead.
pub(super) const HEAD_OWN: (&str, Rule) = ("tie_word_embeddings", Only("[false]"));

/// `tie_word_embeddings` of a map whose model may tie its head to the embedding: read by
/// [`tied`], and held against the folder's head by [`check_head`].
pub(super) const HEAD_EITHER: (&str, Rule) = ("tie_word_embeddings", Read);

/// `tie_word_embeddings`, which must be there and a bool.
pub(super) fn tied<E: From<String>>(c: &Section<'_, E>) -> Result<bool, E> {
    c.field("tie_word_embeddings")?
        .as_bool()
        .ok_or_else(|| c.bad("tie_word_embeddings", "must be a bool"))
}

/// The flag `tied` held against the head the folder holds (its name, `None` for none): a tied
/// model has no head beside the embedding, and one that is not tied has one. Either mismatch is
/// refused, so no folder is read as the head it does not hold.
pub(super) fn check_head(tied: bool, head: Option<&str>) -> Result<(), String> {
    match (tied, head) {
        (true, Some(name)) => Err(format!(
            "{name} is a head beside the embedding, where config.json.tie_word_embeddings is true"
        )),
        (false, None) => Err(
            "config.json.tie_word_embeddings is false, and the folder holds no head".to_string(),
        ),
        _ => Ok(()),
    }
}

/// `dtype`, which names no tensor's format: each tensor keeps its stored one.
pub(super) const DTYPE: (&str, Rule) = (
    "dtype",
    Ignore(
        Type::String,
        "the dtype HF loads in; each tensor keeps its stored dtype and format",
    ),
);

/// The `config.json` keys of the text model that every trunk map spells alike.
pub(super) const KEYS: &Fields = &[
    ("num_hidden_layers", Read),
    ("max_position_embeddings", Read),
    ("hidden_size", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("head_dim", Read),
    ("rms_norm_eps", Read),
    // hf::vocab::Config's.
    ("vocab_size", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("unk_token_id", Read),
    ("sep_token_id", Read),
    // The feed-forward block's SwiGLU, routed or dense.
    ("hidden_act", Only(r#"["silu"]"#)),
    ("attention_bias", Only("[false]")),
    HEAD_OWN,
    DTYPE,
    (
        "attention_dropout",
        Ignore(Type::Number, "dropout acts in training only"),
    ),
    (
        "initializer_range",
        Ignore(Type::Number, "initializes weights the checkpoint holds"),
    ),
    (
        "router_aux_loss_coef",
        Ignore(Type::Number, "weighs a training loss"),
    ),
    (
        "output_router_logits",
        Ignore(
            Type::Bool,
            "returns the router logits beside an unchanged output",
        ),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
];

/// The counts of [`KEYS`] every trunk map reads: each a positive u32.
pub(super) const COUNTS: &[&str] = &[
    "num_hidden_layers",
    "max_position_embeddings",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
];

/// The feed-forward block a trunk map's layers hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ffn {
    /// A router over `num_experts` experts, `num_experts_per_tok` of them kept a token, each of
    /// `moe_intermediate_size`.
    Routed,
    /// One SwiGLU block of `intermediate_size`: `mlp.{gate,up,down}_proj`, the program's
    /// `ffn_{gate,up,down}`.
    Dense,
}

/// The `config.json` keys of a routed block, beside [`KEYS`].
const ROUTED_KEYS: &Fields = &[
    ("num_experts", Read),
    ("num_experts_per_tok", Read),
    ("moe_intermediate_size", Read),
];

/// The `config.json` key of a dense block, beside [`KEYS`].
const DENSE_KEYS: &Fields = &[("intermediate_size", Read)];

impl Ffn {
    /// The `config.json` keys of the block beside [`KEYS`].
    pub fn keys(self) -> &'static Fields {
        match self {
            Ffn::Routed => ROUTED_KEYS,
            Ffn::Dense => DENSE_KEYS,
        }
    }

    /// The counts of [`Ffn::keys`]: each a positive u32.
    pub fn counts(self) -> &'static [&'static str] {
        match self {
            Ffn::Routed => &[
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
            ],
            Ffn::Dense => &["intermediate_size"],
        }
    }
}

/// The counts [`COUNTS`] and [`Ffn::counts`] name hold together: the query heads group by the
/// key heads, and a routed token keeps no more experts than the layer routes over. `n` is a count
/// by its key.
pub(super) fn check_counts<E: From<String>>(
    c: &Section<'_, E>,
    ffn: Ffn,
    n: impl Fn(&str) -> u32,
) -> Result<(), E> {
    if !n("num_attention_heads").is_multiple_of(n("num_key_value_heads")) {
        return Err(c.bad("num_attention_heads", "must group by num_key_value_heads"));
    }
    if ffn == Ffn::Routed && n("num_experts_per_tok") > n("num_experts") {
        return Err(c.bad("num_experts_per_tok", "exceeds num_experts"));
    }
    Ok(())
}

/// The header keys, under the architecture's name, that the counts and two numbers of the config
/// make for every trunk map: the layers and the context, the widths, the heads, the rope's base
/// and the norms' epsilon, and the block's: the experts a routed layer routes over and a token
/// keeps with their width, or a dense layer's width.
pub(super) fn metadata(
    n: impl Fn(&str) -> u32,
    ffn: Ffn,
    rope_base: f32,
    rms_eps: f32,
) -> Vec<(String, Value)> {
    let count = |key: &str, from: &str| (key.to_string(), Value::U32(n(from)));
    let mut keys = vec![
        count("block_count", "num_hidden_layers"),
        count("context_length", "max_position_embeddings"),
        count("embedding_length", "hidden_size"),
        count("attention.head_count", "num_attention_heads"),
        count("attention.head_count_kv", "num_key_value_heads"),
        ("rope.freq_base".to_string(), Value::F32(rope_base)),
        (
            "attention.layer_norm_rms_epsilon".to_string(),
            Value::F32(rms_eps),
        ),
    ];
    if ffn == Ffn::Routed {
        keys.extend([
            count("expert_count", "num_experts"),
            count("expert_used_count", "num_experts_per_tok"),
        ]);
    }
    keys.extend([
        count("attention.key_length", "head_dim"),
        count("attention.value_length", "head_dim"),
    ]);
    keys.push(match ffn {
        Ffn::Routed => count("expert_feed_forward_length", "moe_intermediate_size"),
        Ffn::Dense => count("feed_forward_length", "intermediate_size"),
    });
    keys
}

/// Whether `shape` names a tensor of at least one value, whose count fits `usize`.
pub(super) fn sized(shape: &[usize]) -> bool {
    !shape.is_empty()
        && !shape.contains(&0)
        && shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .is_some()
}

/// The program's attention projection of the checkpoint tensor `rest` (the name past `layers.N.`)
/// of layer `l`, with the role it is read in: the query, key, value and output.
pub(super) fn projections(l: usize, rest: &str) -> Option<(String, Plain)> {
    Some(match rest {
        "self_attn.q_proj.weight" => (common::attn_q(l), Plain::Q8_0),
        "self_attn.k_proj.weight" => (common::attn_k(l), Plain::Q8_0),
        "self_attn.v_proj.weight" => (common::attn_v(l), Plain::Q8_0),
        "self_attn.o_proj.weight" => (common::attn_output(l), Plain::Q8_0),
        _ => return None,
    })
}

/// The program's per-head norm of the checkpoint tensor `rest` of layer `l`: the query's and the
/// key's.
pub(super) fn head_norms(l: usize, rest: &str) -> Option<(String, Plain)> {
    Some(match rest {
        "self_attn.q_norm.weight" => (common::attn_q_norm(l), Plain::F32),
        "self_attn.k_norm.weight" => (common::attn_k_norm(l), Plain::F32),
        _ => return None,
    })
}

/// The program's attention tensor of the checkpoint tensor `rest` of layer `l`, with the role it
/// is read in: the four projections ([`projections`]) and the per-head norms ([`head_norms`]).
pub(super) fn attention(l: usize, rest: &str) -> Option<(String, Plain)> {
    projections(l, rest).or_else(|| head_norms(l, rest))
}

/// The program's norm of the checkpoint tensor `rest` of layer `l`: the one before attention and
/// the one before the feed-forward block, which `ffn_norm` names.
pub(super) fn block_norms(
    l: usize,
    rest: &str,
    ffn_norm: fn(usize) -> String,
) -> Option<(String, Plain)> {
    Some(match rest {
        "input_layernorm.weight" => (common::attn_norm(l), Plain::F32),
        "post_attention_layernorm.weight" => (ffn_norm(l), Plain::F32),
        _ => return None,
    })
}

/// The program's router of the checkpoint tensor `rest` of layer `l`.
pub(super) fn router(l: usize, rest: &str) -> Option<(String, Plain)> {
    (rest == "mlp.gate.weight").then(|| (common::ffn_gate_inp(l), Plain::Bf16))
}

/// The program's dense block of the checkpoint tensor `rest` of layer `l`: the SwiGLU's gate, up
/// and down projections.
pub(super) fn dense_ffn(l: usize, rest: &str) -> Option<(String, Plain)> {
    Some(match rest {
        "mlp.gate_proj.weight" => (common::ffn_gate(l), Plain::Q8_0),
        "mlp.up_proj.weight" => (common::ffn_up(l), Plain::Q8_0),
        "mlp.down_proj.weight" => (common::ffn_down(l), Plain::Q8_0),
        _ => return None,
    })
}

/// The program's routed experts of the checkpoint tensor `rest` of layer `l` when the folder
/// stacks them, one `switch_mlp` tensor of `experts` experts a projection; `None` for a tensor
/// that is not in `mlp.switch_mlp`. A projection the program has no stack for, and a stack of
/// another shape than `[experts, rows, columns]`, are refused with why.
pub(super) fn stacked_experts(
    l: usize,
    rest: &str,
    shape: &[usize],
    experts: usize,
) -> Option<Result<(String, Plain), String>> {
    let proj = rest.strip_prefix("mlp.switch_mlp.")?;
    let to = match proj {
        "gate_proj.weight" => common::ffn_gate_exps(l),
        "up_proj.weight" => common::ffn_up_exps(l),
        "down_proj.weight" => common::ffn_down_exps(l),
        _ => return Some(Err("not a routed expert projection".to_string())),
    };
    Some(match *shape {
        [e, _, _] if e == experts => Ok((to, Plain::Q8_0)),
        _ => Err(format!(
            "{shape:?} is not [{experts} experts, rows, columns]"
        )),
    })
}

/// The program's tensor of the model-level checkpoint tensor `short` (the name past the text
/// prefix): the token embedding and the output norm.
pub(super) fn global(short: &str) -> Option<(String, Plain)> {
    Some(match short {
        "embed_tokens.weight" => (common::token_embd(), Plain::Q8_0),
        "norm.weight" => (common::output_norm(), Plain::F32),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name the trunk maps takes the program's name and the role its tensor is read in; a
    /// name it does not own is `None`, so a map's own arm decides it.
    #[test]
    fn the_trunk_names_what_every_map_names_alike() {
        let ffn = |l| format!("blk.{l}.ffn_norm.weight");
        for (rest, want) in [
            (
                "self_attn.q_proj.weight",
                ("blk.3.attn_q.weight", Plain::Q8_0),
            ),
            (
                "self_attn.k_proj.weight",
                ("blk.3.attn_k.weight", Plain::Q8_0),
            ),
            (
                "self_attn.v_proj.weight",
                ("blk.3.attn_v.weight", Plain::Q8_0),
            ),
            (
                "self_attn.o_proj.weight",
                ("blk.3.attn_output.weight", Plain::Q8_0),
            ),
            (
                "self_attn.q_norm.weight",
                ("blk.3.attn_q_norm.weight", Plain::F32),
            ),
            (
                "self_attn.k_norm.weight",
                ("blk.3.attn_k_norm.weight", Plain::F32),
            ),
        ] {
            assert_eq!(
                attention(3, rest),
                Some((want.0.to_string(), want.1)),
                "{rest}"
            );
        }
        // A model with no head norms reads the projections alone.
        for rest in ["self_attn.q_proj.weight", "self_attn.o_proj.weight"] {
            assert_eq!(projections(3, rest), attention(3, rest), "{rest}");
            assert_eq!(head_norms(3, rest), None, "{rest}");
        }
        for rest in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            assert_eq!(head_norms(3, rest), attention(3, rest), "{rest}");
            assert_eq!(projections(3, rest), None, "{rest}");
        }
        assert_eq!(
            block_norms(3, "input_layernorm.weight", ffn),
            Some(("blk.3.attn_norm.weight".to_string(), Plain::F32))
        );
        assert_eq!(
            block_norms(3, "post_attention_layernorm.weight", ffn),
            Some(("blk.3.ffn_norm.weight".to_string(), Plain::F32))
        );
        assert_eq!(
            router(3, "mlp.gate.weight"),
            Some(("blk.3.ffn_gate_inp.weight".to_string(), Plain::Bf16))
        );
        for (rest, to) in [
            ("mlp.gate_proj.weight", "blk.3.ffn_gate.weight"),
            ("mlp.up_proj.weight", "blk.3.ffn_up.weight"),
            ("mlp.down_proj.weight", "blk.3.ffn_down.weight"),
        ] {
            assert_eq!(
                dense_ffn(3, rest),
                Some((to.to_string(), Plain::Q8_0)),
                "{rest}"
            );
            assert_eq!(router(3, rest), None, "{rest}");
        }
        assert_eq!(
            global("embed_tokens.weight"),
            Some(("token_embd.weight".to_string(), Plain::Q8_0))
        );
        assert_eq!(
            global("norm.weight"),
            Some(("output_norm.weight".to_string(), Plain::F32))
        );
        for rest in [
            "self_attn.q_proj.bias",
            "self_attn.indexer.wq_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "mlp.shared_expert_gate.weight",
            "mlp.gate.bias",
            "mlp.gate_proj.bias",
        ] {
            assert_eq!(attention(3, rest), None, "{rest}");
            assert_eq!(block_norms(3, rest, ffn), None, "{rest}");
            assert_eq!(router(3, rest), None, "{rest}");
            assert_eq!(dense_ffn(3, rest), None, "{rest}");
            assert!(stacked_experts(3, rest, &[8, 4, 4], 8).is_none(), "{rest}");
        }
        assert_eq!(global("layers.0.mlp.gate.weight"), None);
        assert_eq!(dense_ffn(3, "mlp.switch_mlp.gate_proj.weight"), None);
    }

    /// The tie flag must be there and a bool, and is held against the folder's head both ways:
    /// a tied model that holds a head and an untied one that holds none are refused by name.
    /// Mutants (each red at its line): `check_head` reading `(true, Some)` as fine (the
    /// first), `(false, None)` as fine (the second), `tied` defaulting a missing flag (the
    /// third).
    #[test]
    fn the_tie_flag_is_held_against_the_head_the_folder_holds() {
        #[derive(Debug, PartialEq)]
        struct Why(String);
        impl From<String> for Why {
            fn from(why: String) -> Why {
                Why(why)
            }
        }
        assert_eq!(check_head(true, None), Ok(()));
        assert_eq!(check_head(false, Some("lm_head.weight")), Ok(()));
        let e = check_head(true, Some("lm_head.weight")).unwrap_err();
        assert!(e.contains("lm_head.weight") && e.contains("is true"), "{e}");
        let e = check_head(false, None).unwrap_err();
        assert!(e.contains("is false") && e.contains("no head"), "{e}");
        let json = serde_json::json!({"tie_word_embeddings": true, "bad": 1});
        let c = Section::<Why>::new(&json, "config.json").unwrap();
        assert_eq!(tied(&c), Ok(true));
        let json = serde_json::json!({"tie_word_embeddings": "yes"});
        let c = Section::<Why>::new(&json, "config.json").unwrap();
        assert_eq!(
            tied(&c),
            Err(Why("config.json.tie_word_embeddings must be a bool".into()))
        );
        let json = serde_json::json!({});
        let c = Section::<Why>::new(&json, "config.json").unwrap();
        assert_eq!(
            tied(&c),
            Err(Why("config.json.tie_word_embeddings is missing".into()))
        );
        // A map whose model has its own head refuses a tied folder, as before.
        assert!(matches!(HEAD_OWN.1, Only("[false]")));
        assert!(matches!(HEAD_EITHER.1, Read));
    }

    /// A shape has values when no dim is zero and their product fits `usize`.
    #[test]
    fn a_shape_with_no_values_or_too_many_is_not_sized() {
        assert!(sized(&[1]));
        assert!(sized(&[4, 8, 2]));
        assert!(!sized(&[]));
        assert!(!sized(&[4, 0]));
        assert!(!sized(&[usize::MAX, 2]));
    }

    /// A stack of the layer's experts takes the projection's program name; its first dim is the
    /// layer's expert count, and a projection or a shape that is not a stack is refused with why.
    #[test]
    fn stacked_experts_are_one_tensor_a_projection() {
        for (proj, to) in [
            ("gate_proj", "blk.2.ffn_gate_exps.weight"),
            ("up_proj", "blk.2.ffn_up_exps.weight"),
            ("down_proj", "blk.2.ffn_down_exps.weight"),
        ] {
            assert_eq!(
                stacked_experts(2, &format!("mlp.switch_mlp.{proj}.weight"), &[8, 6, 4], 8),
                Some(Ok((to.to_string(), Plain::Q8_0)))
            );
        }
        assert_eq!(
            stacked_experts(2, "mlp.switch_mlp.gate_proj.weight", &[7, 6, 4], 8),
            Some(Err(
                "[7, 6, 4] is not [8 experts, rows, columns]".to_string()
            ))
        );
        assert_eq!(
            stacked_experts(2, "mlp.switch_mlp.gate_proj.weight", &[8, 6], 8),
            Some(Err("[8, 6] is not [8 experts, rows, columns]".to_string()))
        );
        assert_eq!(
            stacked_experts(2, "mlp.switch_mlp.gate_proj.bias", &[8, 6, 4], 8),
            Some(Err("not a routed expert projection".to_string()))
        );
    }

    /// The counts' checks: heads that do not group, and a token that keeps more experts than the
    /// layer has.
    #[test]
    fn counts_that_do_not_hold_together_are_refused() {
        #[derive(Debug, PartialEq)]
        struct Why(String);
        impl From<String> for Why {
            fn from(why: String) -> Why {
                Why(why)
            }
        }
        let json = serde_json::json!({});
        let c = Section::<Why>::new(&json, "config.json").unwrap();
        let with = |heads: u32, kv: u32, experts: u32, used: u32| {
            check_counts(&c, Ffn::Routed, |key| match key {
                "num_attention_heads" => heads,
                "num_key_value_heads" => kv,
                "num_experts" => experts,
                "num_experts_per_tok" => used,
                other => panic!("{other}"),
            })
        };
        // A dense block has no experts to count: asking for one panics.
        let dense = |heads: u32, kv: u32| {
            check_counts(&c, Ffn::Dense, |key| match key {
                "num_attention_heads" => heads,
                "num_key_value_heads" => kv,
                other => panic!("{other}"),
            })
        };
        assert_eq!(dense(24, 4), Ok(()));
        assert_eq!(
            dense(24, 5),
            Err(Why(
                "config.json.num_attention_heads must group by num_key_value_heads".into()
            ))
        );
        assert_eq!(with(32, 4, 128, 8), Ok(()));
        assert_eq!(
            with(32, 5, 128, 8),
            Err(Why(
                "config.json.num_attention_heads must group by num_key_value_heads".into()
            ))
        );
        assert_eq!(
            with(32, 4, 8, 9),
            Err(Why(
                "config.json.num_experts_per_tok exceeds num_experts".into()
            ))
        );
    }

    /// The header keys are the counts under the names the program's readers use, and the rope
    /// base and epsilon as given.
    #[test]
    fn the_header_keys_are_the_counts_by_their_program_names() {
        let keys = metadata(
            |key| match key {
                "num_hidden_layers" => 48,
                "max_position_embeddings" => 262_144,
                "hidden_size" => 2048,
                "num_attention_heads" => 32,
                "num_key_value_heads" => 4,
                "num_experts" => 128,
                "num_experts_per_tok" => 8,
                "head_dim" => 128,
                "moe_intermediate_size" => 768,
                other => panic!("{other}"),
            },
            Ffn::Routed,
            1.0e7,
            1.0e-6,
        );
        let get = |key: &str| keys.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert_eq!(keys.len(), 12);
        assert_eq!(get("block_count"), Some(Value::U32(48)));
        assert_eq!(get("attention.head_count_kv"), Some(Value::U32(4)));
        assert_eq!(get("expert_feed_forward_length"), Some(Value::U32(768)));
        assert_eq!(get("attention.key_length"), Some(Value::U32(128)));
        assert_eq!(get("attention.value_length"), Some(Value::U32(128)));
        assert_eq!(get("rope.freq_base"), Some(Value::F32(1.0e7)));
        assert_eq!(
            get("attention.layer_norm_rms_epsilon"),
            Some(Value::F32(1.0e-6))
        );
        assert_eq!(COUNTS.len() + Ffn::Routed.counts().len(), 9);
    }

    /// A dense block's header keys are the common ones and its width as `feed_forward_length`:
    /// no expert key, and no count of one is asked for.
    #[test]
    fn a_dense_block_makes_its_width_key_and_no_expert_key() {
        let keys = metadata(
            |key| match key {
                "num_hidden_layers" => 64,
                "max_position_embeddings" => 262_144,
                "hidden_size" => 5120,
                "num_attention_heads" => 24,
                "num_key_value_heads" => 4,
                "head_dim" => 256,
                "intermediate_size" => 17_408,
                other => panic!("{other}"),
            },
            Ffn::Dense,
            1.0e7,
            1.0e-6,
        );
        let get = |key: &str| keys.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert_eq!(keys.len(), 10);
        assert_eq!(get("feed_forward_length"), Some(Value::U32(17_408)));
        assert_eq!(get("block_count"), Some(Value::U32(64)));
        assert_eq!(get("attention.key_length"), Some(Value::U32(256)));
        for key in [
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
        ] {
            assert_eq!(get(key), None, "{key}");
        }
        assert_eq!(COUNTS.len() + Ffn::Dense.counts().len(), 7);
    }

    /// Every count a table names is a key the table reads, so a count the maps ask for is one
    /// the schema lets through.
    #[test]
    fn every_count_is_a_read_key_of_its_table() {
        for (keys, counts) in [
            (KEYS, COUNTS),
            (Ffn::Routed.keys(), Ffn::Routed.counts()),
            (Ffn::Dense.keys(), Ffn::Dense.counts()),
        ] {
            for count in counts {
                assert!(
                    keys.iter()
                        .any(|(key, rule)| key == count && matches!(rule, Read)),
                    "{count}"
                );
            }
        }
    }
}
