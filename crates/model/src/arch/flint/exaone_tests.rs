//! The generalized reader on K-EXAONE-2.0's own values: a 79-block synthetic
//! header built from `hf/config.json`'s arrays, read as the `exaone-moe`
//! description row states it, and the refusals of the facts the row owns.

use super::spec::read;
use crate::arch::synthetic::{V, header_shaped};
use crate::placement::PlacementError;
use models::{Act, Ffn, Mixer, Need, RopeMode, RouterNorm};

/// The trunk's 78 blocks and the next-token one.
const BLOCKS: usize = 79;
const TRUNK: usize = 78;
/// The full-attention blocks: 0 and every fourth from 5 (the NoPE layers).
const FULL: [usize; 20] = [
    0, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77,
];
/// The first clamped block; 62..=77 carry the 7.0 limit.
const CLAMP_FROM: usize = 62;

/// The config's `sliding_windows` array (0: full attention), entry 78 the
/// next-token layer's 128.
fn window_values() -> Vec<i32> {
    (0..BLOCKS)
        .map(|l| {
            if FULL.contains(&l) {
                0
            } else if l == 1 {
                4096
            } else {
                128
            }
        })
        .collect()
}

/// The config's clamp limits, 7.0 on 62..=77.
fn clamp_values() -> Vec<f32> {
    (0..BLOCKS)
        .map(|l| {
            if (CLAMP_FROM..TRUNK).contains(&l) {
                7.0
            } else {
                0.0
            }
        })
        .collect()
}

/// A template whose markup `ToolFormat::of_template` and
/// `ReasoningFormat::of_template` read: the same markers the 2.0 folder's
/// template teaches.
const TEMPLATE: &str = "<|assistant|>\n<tool_call>\n<function=name>\n<parameter=arg>\nvalue\n</parameter>\n</function>\n</tool_call>\n<think>\n\n</think>\n";

/// The keys of the 2.0 config, in the GGUF names the reader reads.
fn keys() -> Vec<(&'static str, V)> {
    vec![
        ("block_count", V::U32(BLOCKS as u32)),
        ("nextn_predict_layers", V::U32(1)),
        ("embedding_length", V::U32(6144)),
        ("context_length", V::U32(262144)),
        ("vocab_size", V::U32(153600)),
        ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
        ("attention.head_count", V::U32(64)),
        ("attention.head_count_kv", V::U32(8)),
        ("attention.key_length", V::U32(128)),
        ("attention.value_length", V::U32(128)),
        ("attention.sliding_window", V::I32s(window_values())),
        (
            "attention.sliding_window_pattern",
            V::Bools(window_values().iter().map(|&w| w > 0).collect()),
        ),
        (
            "attention.rope_pattern",
            V::Bools(window_values().iter().map(|&w| w > 0).collect()),
        ),
        ("rope.freq_base", V::F32(1e6)),
        ("rope.dimension_count", V::U32(128)),
        ("leading_dense_block_count", V::U32(2)),
        ("feed_forward_length", V::U32(18432)),
        ("expert_feed_forward_length", V::U32(2048)),
        ("expert_count", V::U32(256)),
        ("expert_used_count", V::U32(8)),
        ("expert_shared_count", V::U32(1)),
        ("expert_shared_feed_forward_length", V::U32(2048)),
        ("expert_weights_norm", V::Bool(true)),
        ("expert_weights_scale", V::F32(2.5)),
        ("expert_gating_func", V::U32(2)),
        ("expert_group_count", V::U32(1)),
        ("expert_group_used_count", V::U32(1)),
        ("swiglu_clamp_exp", V::F32s(clamp_values())),
        ("swiglu_clamp_shexp", V::F32s(clamp_values())),
    ]
}

/// Every tensor of the 2.0 shapes, the dense blocks (the lead and the
/// next-token layer) with the dense FFN and the routed ones with the stacks.
fn tensors() -> Vec<(String, Vec<u64>)> {
    let (e, v) = (6144_u64, 153600_u64);
    let (q, kv, d) = (64 * 128, 8 * 128, 128_u64);
    let mut out = vec![
        ("token_embd.weight".into(), vec![e, v]),
        ("output.weight".into(), vec![e, v]),
        ("output_norm.weight".into(), vec![e]),
    ];
    for l in 0..BLOCKS {
        let dense = !(2..TRUNK).contains(&l);
        let mut layer: Vec<(&str, Vec<u64>)> = vec![
            ("attn_norm.weight", vec![e]),
            ("attn_q.weight", vec![e, q]),
            ("attn_k.weight", vec![e, kv]),
            ("attn_v.weight", vec![e, kv]),
            ("attn_output.weight", vec![q, e]),
            ("attn_q_norm.weight", vec![d]),
            ("attn_k_norm.weight", vec![d]),
            ("ffn_norm.weight", vec![e]),
        ];
        if dense {
            layer.extend([
                ("ffn_gate.weight", vec![e, 18432]),
                ("ffn_up.weight", vec![e, 18432]),
                ("ffn_down.weight", vec![18432, e]),
            ]);
        } else {
            layer.extend([
                ("ffn_gate_inp.weight", vec![e, 256]),
                ("exp_probs_b.bias", vec![256]),
                ("ffn_gate_exps.weight", vec![e, 2048, 256]),
                ("ffn_up_exps.weight", vec![e, 2048, 256]),
                ("ffn_down_exps.weight", vec![2048, e, 256]),
                ("ffn_gate_shexp.weight", vec![e, 2048]),
                ("ffn_up_shexp.weight", vec![e, 2048]),
                ("ffn_down_shexp.weight", vec![2048, e]),
            ]);
        }
        if l >= TRUNK {
            layer.extend([
                ("nextn.eh_proj.weight", vec![2 * e, e]),
                ("nextn.enorm.weight", vec![e]),
                ("nextn.hnorm.weight", vec![e]),
                ("nextn.shared_head_norm.weight", vec![e]),
            ]);
        }
        out.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
    }
    out
}

/// The read of a shaped `exaone-moe` header of `kv` and `ts`; the file is removed.
fn check(
    tag: &str,
    kv: &[(&str, V)],
    ts: &[(String, Vec<u64>)],
) -> Result<crate::arch::Read, PlacementError> {
    check_arch(tag, "exaone-moe", kv, ts)
}

/// [`check`] under another architecture string.
fn check_arch(
    tag: &str,
    arch: &str,
    kv: &[(&str, V)],
    ts: &[(String, Vec<u64>)],
) -> Result<crate::arch::Read, PlacementError> {
    let path = header_shaped(
        &format!("exaone-{tag}"),
        arch,
        kv,
        &[
            ("tokenizer.ggml.pre", V::Str("exaone-moe")),
            ("tokenizer.chat_template", V::Str(TEMPLATE)),
        ],
        ts,
    );
    let split = gguf::Split::open(&path).unwrap();
    let result = read(&split);
    std::fs::remove_file(path).unwrap();
    result
}

/// The 79-block header reads to the config's description: full and windowed
/// attention exactly where the arrays say, rope where windowed, the dense lead
/// and the next-token layer, the clamp on 62..=77's routed and shared experts,
/// the biased sigmoid router, no sinks, and the MTP layer in `spec.mtp`.
#[test]
fn the_exaone_header_reads_layer_by_layer() {
    let r = check("description", &keys(), &tensors()).unwrap();
    let s = &r.spec;
    assert_eq!(s.arch, models::Arch::ExaoneMoe);
    assert_eq!((s.hidden, s.vocab, s.ctx_train), (6144, 153600, 262144));
    assert_eq!(s.rms_eps.to_bits(), 1e-5_f32.to_bits());
    assert_eq!(s.layers.len(), TRUNK);
    assert_eq!(s.mtp.len(), 1);
    assert_eq!(s.chat.pre, "exaone-moe");
    assert_eq!(s.chat.tools, Some(models::ToolFormat::QwenXml));
    assert_eq!(s.chat.reasoning, Some(models::ReasoningFormat::THINK));
    let want_window = |l: usize| {
        if FULL.contains(&l) {
            None
        } else if l == 1 {
            Some(4096)
        } else {
            Some(128)
        }
    };
    for (l, layer) in s.layers.iter().chain(&s.mtp).enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("layer {l} not GQA");
        };
        assert_eq!(
            (g.heads, g.kv_heads, g.head_dim, g.value_dim),
            (64, 8, 128, 128)
        );
        assert_eq!(g.window, want_window(l), "layer {l}");
        let roped = want_window(l).is_some();
        assert_eq!(
            g.rope.mode,
            if roped {
                RopeMode::Neox
            } else {
                RopeMode::None
            }
        );
        assert_eq!(g.rope.dims, if roped { 128 } else { 0 });
        assert_eq!(g.rope.base, if roped { 1e6 } else { 0.0 });
        assert!(g.qk_norm && !g.sinks, "layer {l}");
        let clamped = (CLAMP_FROM..TRUNK).contains(&l);
        let want = if clamped {
            Act::SwiGluGateFirst { limit: 7.0 }
        } else {
            Act::SwiGlu { limit: None }
        };
        let mtp_layer = l >= TRUNK;
        match &layer.ffn {
            Ffn::Dense { ff, act } => {
                assert!(l < 2 || mtp_layer, "layer {l} dense");
                assert_eq!(*ff, 18432);
                assert_eq!(*act, Act::SwiGlu { limit: None }, "layer {l}");
            }
            Ffn::Moe(m) => {
                assert!((2..TRUNK).contains(&l), "layer {l} routed");
                assert_eq!((m.experts, m.top_k, m.expert_ff), (256, 8, 2048));
                assert_eq!(m.act, want, "layer {l}");
                assert_eq!(
                    m.router,
                    models::Router {
                        score: models::Score::Sigmoid,
                        bias: true,
                        norm: RouterNorm::SelectedSum,
                        scale: 2.5,
                        hash: false,
                    },
                    "layer {l}"
                );
                let shared = m.shared.as_ref().unwrap();
                assert_eq!(shared.ff, 2048);
                assert_eq!(shared.act, want, "layer {l}");
                assert!(!shared.sigmoid_gate);
            }
        }
    }
    let needs = models::needs(s);
    assert_eq!(
        needs
            .iter()
            .filter(|(n, _)| matches!(n, Need::ExpertAct { .. }))
            .count(),
        32
    );
    assert!(needs.iter().any(|(n, _)| matches!(
        n,
        Need::Router(m) if m.experts == 256 && m.top_k == 8
    )));
}

/// The defaults the read records: the row's consumed facts and the one key it
/// filled, each with its source, and nothing else.
#[test]
fn the_defaults_name_the_row_and_the_fill() {
    let r = check("defaults", &keys(), &tensors()).unwrap();
    assert_eq!(
        r.defaults,
        [
            "router norm = selected-sum (LG modeling_exaone_moe.py:269-270: the kept weights over their sum + 1e-20)",
            "swiglu_clamp = 0 (llama.cpp's fill, llama-model.cpp:1555-1557)",
            "swiglu clamp = gate-first (LG modeling_exaone_moe.py:227-232: silu(min(g, L))·clamp(u, ±L))",
            "rope pairs = neox (llama.cpp src/llama-model.cpp:3799,3834: EXAONE's rope type is NEOX)",
        ]
    );
}

/// Without `attention.rope_pattern` the row's fallback ropes a layer exactly
/// where it is windowed, and the read records the rule it took.
#[test]
fn an_absent_rope_pattern_takes_the_rows_fallback() {
    let mut kv = keys();
    kv.retain(|(k, _)| *k != "attention.rope_pattern");
    let r = check("rope-fallback", &kv, &tensors()).unwrap();
    for (l, layer) in r.spec.layers.iter().chain(&r.spec.mtp).enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("layer {l} not GQA");
        };
        let windowed = !FULL.contains(&l);
        assert_eq!(
            g.rope.mode,
            if windowed {
                RopeMode::Neox
            } else {
                RopeMode::None
            }
        );
    }
    assert!(r
        .defaults
        .iter()
        .any(|d| d == "rope = where windowed (llama.cpp src/models/exaone-moe.cpp:7: rope_pattern is is_swa_impl)"));
}

/// A window array shorter than the block count is refused by name. Mutant: the
/// array read told the layer count is the array's own length.
#[test]
fn a_short_window_array_is_refused_by_name() {
    let mut kv = keys();
    let short = window_values()[..TRUNK].to_vec();
    kv.iter_mut()
        .find(|(k, _)| *k == "attention.sliding_window")
        .unwrap()
        .1 = V::I32s(short);
    match check("window-length", &kv, &tensors()) {
        Err(PlacementError::Metadata { key, detail }) => {
            assert_eq!(key, "exaone-moe.attention.sliding_window");
            assert!(detail.contains("has 78 values for 79 layers"), "{detail}");
        }
        other => panic!("the length is refused, got {other:?}"),
    }
}

/// A `sliding_window_pattern` that disagrees with the window array at one
/// layer is refused naming the layer and both values. Mutant: the agreement
/// check's predicate `false`.
#[test]
fn a_pattern_disagreeing_with_the_window_array_is_refused() {
    let mut kv = keys();
    let mut pattern = window_values().iter().map(|&w| w > 0).collect::<Vec<_>>();
    pattern[2] = false;
    kv.iter_mut()
        .find(|(k, _)| *k == "attention.sliding_window_pattern")
        .unwrap()
        .1 = V::Bools(pattern);
    match check("pattern-agree", &kv, &tensors()) {
        Err(PlacementError::Metadata { key, detail }) => {
            assert_eq!(key, "exaone-moe.attention.sliding_window_pattern");
            assert!(detail.contains("layer 2"), "{detail}");
            assert!(detail.contains("128"), "{detail}");
        }
        other => panic!("the disagreement is refused, got {other:?}"),
    }
}

/// A non-zero clamp limit under a row whose clamp order is none (the same
/// header under the `flint` string) is refused naming the key. Mutant: the
/// order check dropped.
#[test]
fn a_clamped_file_under_a_clampless_row_is_refused() {
    match check_arch("flint-clamp", "flint", &keys(), &tensors()) {
        Err(PlacementError::Metadata { key, detail }) => {
            assert_eq!(key, "flint.swiglu_clamp_exp");
            assert!(detail.contains("7 at layer 62"), "{detail}");
            assert!(detail.contains("clamp order is none"), "{detail}");
        }
        other => panic!("the clamp is refused, got {other:?}"),
    }
}

/// An architecture string no description row states is refused before any key
/// is read. Mutant: the row lookup falls back to the exaone-moe row.
#[test]
fn an_architecture_no_row_states_is_refused() {
    let path = header_shaped(
        "exaone-no-row",
        "exaone-test",
        &[],
        &[("tokenizer.ggml.pre", V::Str("exaone-moe"))],
        &[],
    );
    let split = gguf::Split::open(&path).unwrap();
    match read(&split) {
        Err(PlacementError::Metadata { key, detail }) => {
            assert_eq!(key, "general.architecture");
            assert!(detail.contains("\"exaone-test\""), "{detail}");
            assert!(detail.contains("no description row"), "{detail}");
        }
        other => panic!("the string is refused, got {other:?}"),
    }
    std::fs::remove_file(path).unwrap();
}
