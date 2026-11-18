use super::spec::read;
use crate::arch::synthetic::{V, header_shaped};
use crate::placement::PlacementError;
use models::{Ffn, Mixer, Need, RopeMode};

pub(super) fn keys() -> Vec<(&'static str, V)> {
    vec![
        ("block_count", V::U32(4)),
        ("embedding_length", V::U32(16)),
        ("context_length", V::U32(128)),
        ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
        ("attention.head_count", V::U32(4)),
        ("attention.head_count_kv", V::U32(2)),
        ("attention.key_length", V::U32(8)),
        ("attention.value_length", V::U32(8)),
        ("rope.freq_base", V::F32(10000.0)),
        ("attention.sliding_window", V::U32(17)),
        (
            "attention.sliding_window_pattern",
            V::Bools(vec![false, true, false, true]),
        ),
        (
            "attention.rope_pattern",
            V::Bools(vec![false, false, true, true]),
        ),
        ("leading_dense_block_count", V::U32(1)),
        ("feed_forward_length", V::U32(24)),
        ("expert_feed_forward_length", V::U32(8)),
        ("expert_count", V::U32(8)),
        ("expert_used_count", V::U32(2)),
        ("expert_shared_count", V::U32(2)),
        ("expert_shared_feed_forward_length", V::U32(16)),
        ("expert_weights_norm", V::Bool(true)),
        ("expert_weights_scale", V::F32(1.0)),
        ("expert_gating_func", V::U32(2)),
    ]
}

pub(super) fn tensors() -> Vec<(String, Vec<u64>)> {
    let mut out = vec![
        ("token_embd.weight".into(), vec![16, 2]),
        ("output.weight".into(), vec![16, 2]),
        ("output_norm.weight".into(), vec![16]),
    ];
    for l in 0..4 {
        let mut layer = vec![
            ("attn_norm.weight", vec![16]),
            ("ffn_norm.weight", vec![16]),
            ("attn_q.weight", vec![16, 32]),
            ("attn_k.weight", vec![16, 16]),
            ("attn_v.weight", vec![16, 16]),
            ("attn_output.weight", vec![32, 16]),
            ("attn_q_norm.weight", vec![8]),
            ("attn_k_norm.weight", vec![8]),
            ("attn_sinks.weight", vec![4]),
        ];
        if l == 0 {
            layer.extend([
                ("ffn_gate.weight", vec![16, 24]),
                ("ffn_up.weight", vec![16, 24]),
                ("ffn_down.weight", vec![24, 16]),
            ]);
        } else {
            layer.extend([
                ("ffn_gate_inp.weight", vec![16, 8]),
                ("ffn_gate_exps.weight", vec![16, 8, 8]),
                ("ffn_up_exps.weight", vec![16, 8, 8]),
                ("ffn_down_exps.weight", vec![8, 16, 8]),
                ("ffn_gate_shexp.weight", vec![16, 16]),
                ("ffn_up_shexp.weight", vec![16, 16]),
                ("ffn_down_shexp.weight", vec![16, 16]),
            ]);
        }
        out.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
    }
    out
}

pub(super) fn check(
    tag: &str,
    keys: &[(&str, V)],
    tensors: &[(String, Vec<u64>)],
) -> Result<crate::arch::Read, PlacementError> {
    let path = header_shaped(
        &format!("flint-{tag}"),
        "flint",
        keys,
        &[("tokenizer.ggml.pre", V::Str("flint"))],
        tensors,
    );
    let split = gguf::Split::open(&path).unwrap();
    let result = read(&split);
    std::fs::remove_file(path).unwrap();
    result
}

#[test]
fn masks_are_independent_and_nope_still_requires_head_norm() {
    let r = check("description", &keys(), &tensors()).unwrap();
    assert_eq!(
        (r.spec.hidden, r.spec.vocab, r.spec.ctx_train),
        (16, 2, 128)
    );
    for (l, layer) in r.spec.layers.iter().enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("not GQA")
        };
        assert_eq!((g.heads, g.kv_heads, g.head_dim, g.value_dim), (4, 2, 8, 8));
        assert_eq!(g.window, (l % 2 == 1).then_some(17));
        assert_eq!(
            g.rope.mode,
            if l < 2 {
                RopeMode::None
            } else {
                RopeMode::Neox
            }
        );
        assert_eq!(g.rope.dims, if l < 2 { 0 } else { 8 });
        assert!(g.qk_norm && g.sinks);
        match &layer.ffn {
            Ffn::Dense { ff, .. } => {
                assert_eq!(l, 0);
                assert_eq!(*ff, 24);
            }
            Ffn::Moe(m) => {
                assert!(l > 0);
                assert_eq!((m.experts, m.top_k, m.expert_ff), (8, 2, 8));
                assert!(!m.router.bias);
                assert_eq!(m.router.norm, models::RouterNorm::SelectedSumFloor);
                let shared = m.shared.as_ref().unwrap();
                assert_eq!(shared.ff, 16);
                assert!(!shared.sigmoid_gate);
            }
        }
    }
    let needs = models::needs(&r.spec);
    assert_eq!(
        needs
            .iter()
            .filter(|(n, _)| matches!(n, Need::QkRope { qk_norm: true, .. }))
            .count(),
        4
    );
    assert_eq!(
        needs
            .iter()
            .filter(|(n, _)| matches!(
                n,
                Need::QkRope {
                    mode: RopeMode::None,
                    dims: 0,
                    ..
                }
            ))
            .count(),
        2
    );
    assert_eq!(
        needs
            .iter()
            .filter(|(n, _)| matches!(n, Need::Gqa { sinks: true, .. }))
            .count(),
        4
    );
}

#[test]
fn metadata_refusals_name_the_invalid_input() {
    for (key, bad) in [
        ("block_count", V::U64(513)),
        ("embedding_length", V::U32(0)),
        ("attention.head_count_kv", V::U32(3)),
        ("attention.head_count_kv", V::I32s(vec![2, 2])),
        ("attention.key_length", V::U64(1 << 63)),
        ("attention.value_length", V::U32(4)),
        ("rope.freq_base", V::F32(f32::NAN)),
        ("attention.layer_norm_rms_epsilon", V::F32(f32::INFINITY)),
        ("attention.sliding_window", V::U32(0)),
        ("leading_dense_block_count", V::U32(5)),
        ("feed_forward_length", V::U32(0)),
        ("expert_feed_forward_length", V::U32(0)),
        ("expert_used_count", V::U32(9)),
        ("expert_shared_count", V::U32(0)),
        ("expert_shared_feed_forward_length", V::U32(8)),
        ("expert_weights_scale", V::F32(f32::NAN)),
        ("expert_weights_scale", V::F32(f32::INFINITY)),
        ("expert_weights_scale", V::F32(f32::NEG_INFINITY)),
        ("expert_weights_scale", V::F32(0.0)),
        ("expert_weights_scale", V::F32(-1.0)),
        ("expert_weights_scale", V::Bool(true)),
        ("expert_weights_norm", V::U32(1)),
        ("expert_gating_func", V::U32(1)),
    ] {
        let mut kv = keys();
        kv.iter_mut().find(|(k, _)| *k == key).unwrap().1 = bad;
        match check(key, &kv, &tensors()) {
            Err(PlacementError::Metadata { key: actual, .. }) => {
                assert_eq!(actual, format!("flint.{key}"))
            }
            Err(other) => panic!("{key}: {other}"),
            Ok(_) => panic!("accepted {key}"),
        }
    }
    for key in ["attention.sliding_window_pattern", "attention.rope_pattern"] {
        for bad in [
            None,
            Some(V::Bool(true)),
            Some(V::Bools(vec![true; 3])),
            Some(V::Bools(vec![true; 5])),
            Some(V::I32s(vec![1, 0, 1, 0])),
        ] {
            let mut kv = keys();
            kv.retain(|(k, _)| *k != key);
            if let Some(value) = bad {
                kv.push((key, value));
            }
            assert!(
                matches!(check(key,&kv,&tensors()),Err(PlacementError::Metadata{key:actual,..}) if actual==format!("flint.{key}"))
            );
        }
    }
}

#[test]
fn router_scale_defaults_only_when_absent() {
    for scale in [None, Some(0.5), Some(2.0)] {
        let mut kv = keys();
        kv.retain(|(k, _)| *k != "expert_weights_scale");
        if let Some(value) = scale {
            kv.push(("expert_weights_scale", V::F32(value)));
        }
        let r = check("scale", &kv, &tensors()).unwrap();
        for layer in &r.spec.layers[1..] {
            let Ffn::Moe(m) = &layer.ffn else {
                panic!("not routed")
            };
            assert_eq!(m.router.scale, scale.unwrap_or(1.0));
        }
        assert_eq!(
            r.defaults
                .iter()
                .filter(|d| d.starts_with("expert_weights_scale"))
                .count(),
            usize::from(scale.is_none())
        );
    }
}

#[test]
fn normalization_policy_is_architecture_specific_and_bias_is_selection_only() {
    for norm in [false, true] {
        for bias in [false, true] {
            let mut kv = keys();
            kv.iter_mut()
                .find(|(k, _)| *k == "expert_weights_norm")
                .unwrap()
                .1 = V::Bool(norm);
            let mut ts = tensors();
            if bias {
                for l in 1..4 {
                    ts.push((format!("blk.{l}.exp_probs_b.bias"), vec![8]));
                }
            }
            let r = check("router-policy", &kv, &ts).unwrap();
            let expected = if norm {
                models::RouterNorm::SelectedSumFloor
            } else {
                models::RouterNorm::None
            };
            let uncovered = crate::arch::coverage::check(&r.spec, &r.tensors);
            for layer in &r.spec.layers[1..] {
                let Ffn::Moe(m) = &layer.ffn else {
                    panic!("not routed")
                };
                assert_eq!(m.router.norm, expected);
                assert_eq!(m.router.bias, bias);
                let shape = models::shape::MoeShape::of(m);
                assert_eq!(shape.rule.norm, expected);
                assert_eq!(shape.rule.bias, bias);
                let refusal = models::shape::select_router(shape).unwrap_err().why;
                if norm {
                    // The widths the table serves this rule at are the table's own: a router row
                    // another family adds at a new width joins the list, and this file's
                    // 8-expert shape stays refused whatever the list holds.
                    let mut served: Vec<u32> = Vec::new();
                    for row in models::shape::ROUTERS.iter().filter(|r| r.runs(shape.rule)) {
                        if !served.contains(&row.experts()) {
                            served.push(row.experts());
                        }
                    }
                    assert!(!served.is_empty() && !served.contains(&shape.experts));
                    assert_eq!(refusal, models::shape::Refusal::RouterWidth { served });
                } else {
                    assert_eq!(refusal, models::shape::Refusal::NoRouterRule);
                }
                assert!(
                    uncovered
                        .iter()
                        .any(|u| u.feature == Need::Router(shape).to_string())
                );
            }
        }
    }
}

#[test]
fn required_shapes_optional_selection_bias_and_unknown_names() {
    let original = tensors();
    for name in [
        "output.weight",
        "blk.0.attn_sinks.weight",
        "blk.1.attn_q_norm.weight",
        "blk.2.ffn_down_shexp.weight",
    ] {
        let mut ts = original.clone();
        ts.retain(|(n, _)| n != name);
        assert!(
            matches!(check("missing",&keys(),&ts),Err(PlacementError::Tensor{name:actual,..}) if actual==name)
        );
    }
    for name in [
        "blk.0.attn_q.weight",
        "blk.1.ffn_gate_exps.weight",
        "blk.2.ffn_down_shexp.weight",
    ] {
        let mut ts = original.clone();
        ts.iter_mut().find(|(n, _)| n == name).unwrap().1[0] += 1;
        assert!(
            matches!(check("shape",&keys(),&ts),Err(PlacementError::Tensor{name:actual,..}) if actual==name)
        );
    }
    let mut ts = original.clone();
    ts.push(("blk.1.exp_probs_b.bias".into(), vec![8]));
    let with = check("bias", &keys(), &ts).unwrap();
    assert!(matches!(&with.spec.layers[1].ffn,Ffn::Moe(m) if m.router.bias));
    ts.last_mut().unwrap().1 = vec![7];
    assert!(
        matches!(check("bad-bias",&keys(),&ts),Err(PlacementError::Tensor{name,..}) if name=="blk.1.exp_probs_b.bias")
    );
    ts = original.clone();
    ts.push(("blk.0.ffn_gate_exps.weight".into(), vec![16, 8, 8]));
    assert!(
        matches!(check("misplaced",&keys(),&ts),Err(PlacementError::Tensor{name,..}) if name=="blk.0.ffn_gate_exps.weight")
    );
    ts = original;
    ts.push(("blk.1.mystery.weight".into(), vec![1]));
    assert!(
        matches!(check("unknown",&keys(),&ts),Err(PlacementError::Unclassified{names}) if names==["blk.1.mystery.weight"])
    );
}
