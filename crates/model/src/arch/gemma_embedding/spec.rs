//! The EmbeddingGemma description: a shared Gemma trunk and two linear maps.

use gguf::{Split, Value};
use models::{Arch, ChatSpec, EncoderSpec, ModelSpec};

use super::{hparams::Hparams, roles};
use crate::arch::{Read, chat_of, spec_u32, spm_pre_default};
use crate::placement::PlacementError;

/// Read the description, tensor roles and architecture-derived defaults.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let key = "tokenizer.ggml.model";
    if split.value(key).and_then(Value::as_str) != Some("llama") {
        return Err(PlacementError::Metadata {
            key: key.into(),
            detail: "must be llama (SentencePiece)".into(),
        });
    }
    let chat = chat_of(split, None, None)?;
    let spec = spec_of(&hp, chat)?;
    let mut defaults = hp.defaults;
    defaults.extend(spm_pre_default(split));
    Ok(Read {
        spec,
        tensors,
        defaults,
    })
}

/// Describe validated hyperparameters without indexing a short per-layer list.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let reach = crate::arch::gemma_encoder::reach(hp.n_layer, hp.kv.len(), &hp.sliding, hp.window)?;
    Ok(ModelSpec {
        arch: Arch::GemmaEmbedding,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer)
            .map(|l| {
                crate::arch::gemma_encoder::layer_spec(
                    hp.n_head,
                    hp.kv[l],
                    hp.attn(l),
                    hp.n_ff,
                    "attention.key_length",
                    "rope.dimension_count",
                )
            })
            .collect::<Result<_, _>>()?,
        encoder: Some(EncoderSpec {
            reach,
            per_layer: None,
            proj: vec![
                spec_u32("dense_2_feat_out", hp.n_dense)?,
                spec_u32("dense_3_feat_out", hp.n_embd)?,
            ],
            embed_scale: true,
            value_norm: false,
            post_norms: true,
            layer_scale: false,
        }),
        mtp: Vec::new(),
        hc: None,
        engram: None,
        logit_epilogue: None,
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::{Act, Ffn, Mixer, Need, RopeMode};

    use super::*;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::arch::{coverage, spec};
    use crate::placement::Role;

    const N: u32 = 24;
    const E: u64 = 32;
    const D: u64 = 8;
    const FF: u64 = 48;
    const DENSE: u64 = 128;

    fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(N)),
            ("embedding_length", V::U32(E as u32)),
            ("context_length", V::U32(2048)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("attention.head_count", V::U32(3)),
            ("attention.head_count_kv", V::U32(1)),
            ("attention.key_length", V::U32(D as u32)),
            ("attention.value_length", V::U32(D as u32)),
            ("rope.dimension_count", V::U32(D as u32)),
            ("rope.freq_base", V::F32(1e6)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.sliding_window", V::U32(11)),
            ("dense_2_feat_in", V::U32(E as u32)),
            ("dense_2_feat_out", V::U32(DENSE as u32)),
            ("dense_3_feat_in", V::U32(DENSE as u32)),
            ("dense_3_feat_out", V::U32(E as u32)),
            ("pooling_type", V::U32(1)),
        ]
    }

    fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut out = vec![
            ("token_embd.weight".into(), vec![E, 2]),
            ("output_norm.weight".into(), vec![E]),
            ("dense_2.weight".into(), vec![E, DENSE]),
            ("dense_3.weight".into(), vec![DENSE, E]),
        ];
        for l in 0..N {
            for (name, dims) in [
                ("attn_norm", vec![E]),
                ("attn_q", vec![E, 3 * D]),
                ("attn_k", vec![E, D]),
                ("attn_v", vec![E, D]),
                ("attn_output", vec![3 * D, E]),
                ("attn_q_norm", vec![D]),
                ("attn_k_norm", vec![D]),
                ("post_attention_norm", vec![E]),
                ("ffn_norm", vec![E]),
                ("ffn_gate", vec![E, FF]),
                ("ffn_up", vec![E, FF]),
                ("ffn_down", vec![FF, E]),
                ("post_ffw_norm", vec![E]),
            ] {
                out.push((format!("blk.{l}.{name}.weight"), dims));
            }
        }
        out
    }

    fn load(
        kv: Vec<(&str, V)>,
        tensors: &[(String, Vec<u64>)],
        tokenizer: &'static str,
    ) -> Result<Read, String> {
        load_with(kv, tensors, &[("tokenizer.ggml.model", V::Str(tokenizer))])
    }

    fn load_with(
        kv: Vec<(&str, V)>,
        tensors: &[(String, Vec<u64>)],
        global: &[(&str, V)],
    ) -> Result<Read, String> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tag = format!(
            "gemma-emb1-{}",
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let path = header_shaped(&tag, "gemma-embedding", &kv, global, tensors);
        let split = Split::open(&path).expect("the synthetic header opens");
        let result = spec(&split).map_err(|e| e.to_string());
        drop(split);
        std::fs::remove_file(&path).expect("remove the synthetic header");
        result
    }

    fn set(kv: &mut [(&str, V)], key: &str, value: V) {
        kv.iter_mut().find(|(k, _)| *k == key).expect(key).1 = value;
    }

    #[test]
    fn the_architecture_supplies_noncausal_attention_and_every_sixth_global_layer() {
        let read = load(keys(), &tensors(), "llama").expect("the header reads without v2 keys");
        let s = &read.spec;
        assert_eq!(s.arch, Arch::GemmaEmbedding);
        assert_eq!(
            crate::arch::Arch::from_name(s.arch.name()).unwrap(),
            crate::arch::Arch::GemmaEmbedding
        );
        assert_eq!((s.hidden, s.vocab, s.ctx_train), (E as u32, 2, 2048));
        let encoder = s.encoder.as_ref().expect("an encoder");
        let full: Vec<_> = encoder
            .reach
            .iter()
            .enumerate()
            .filter_map(|(l, r)| r.is_none().then_some(l))
            .collect();
        assert_eq!(full, [5, 11, 17, 23]);
        assert_eq!(encoder.per_layer, None);
        assert_eq!(encoder.proj, [DENSE as u32, E as u32]);
        assert!(encoder.embed_scale && encoder.post_norms);
        assert!(!encoder.value_norm && !encoder.layer_scale);
        for (l, layer) in s.layers.iter().enumerate() {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("layer {l} is not GQA");
            };
            assert_eq!(
                (g.heads, g.kv_heads, g.head_dim, g.value_dim),
                (3, 1, D as u32, D as u32)
            );
            assert_eq!((g.rope.mode, g.rope.dims), (RopeMode::Neox, D as u32));
            assert_eq!(g.rope.base, if full.contains(&l) { 1e6 } else { 1e4 });
            assert!(g.qk_norm && g.window.is_none());
            assert_eq!(
                layer.ffn,
                Ffn::Dense {
                    ff: FF as u32,
                    act: Act::GeGlu
                }
            );
            assert_eq!(encoder.reach[l], (!full.contains(&l)).then_some(5));
        }
        assert!(
            read.defaults
                .iter()
                .any(|s| s.contains("attention.causal = false"))
        );
        assert!(
            read.defaults
                .iter()
                .any(|s| s.contains("load_swa_pattern(ml, 6)"))
        );
        assert!(s.chat.tools.is_none() && s.chat.reasoning.is_none());
        assert_eq!(s.chat.pre, "llama");
        for name in ["dense_2.weight", "dense_3.weight"] {
            let tensor = read
                .tensors
                .tensors
                .iter()
                .find(|t| t.name == name)
                .expect(name);
            assert_eq!((tensor.role, tensor.layer), (Role::Head, None));
        }
        let needs = models::needs(s);
        assert!(needs.iter().any(|(n, _)| matches!(
            n,
            Need::Bidirectional {
                group: 3,
                reach: Some(5),
                ..
            }
        )));
        let items = coverage::check(s, &read.tensors);
        assert!(items.iter().any(|u| u.feature.contains("bidirectional")));
    }

    #[test]
    fn the_dense_head_and_three_head_trunk_have_one_program() {
        let mut read = load(keys(), &tensors(), "llama").unwrap();
        read.spec.hidden = 768;
        for layer in &mut read.spec.layers {
            let Mixer::Gqa(g) = &mut layer.mixer else {
                panic!("not GQA")
            };
            g.head_dim = 256;
            g.value_dim = 256;
            g.rope.dims = 256;
            layer.ffn = Ffn::Dense {
                ff: 1152,
                act: Act::GeGlu,
            };
        }
        read.spec.encoder.as_mut().unwrap().proj = vec![3072, 768];
        for t in &mut read.tensors.tensors {
            if t.dims.len() >= 2 {
                t.ty = gguf::GgmlType::Q8_0;
            }
        }
        assert!(coverage::check(&read.spec, &read.tensors).is_empty());
        read.spec.encoder.as_mut().unwrap().proj.reverse();
        assert!(!coverage::check(&read.spec, &read.tensors).is_empty());
    }

    #[test]
    fn key_head_counts_accept_scalars_and_per_layer_arrays() {
        for (value, heads) in [
            (V::U32(1), vec![1; N as usize]),
            (V::I32s(vec![1; N as usize]), vec![1; N as usize]),
            (
                V::I32s((0..N).map(|l| if l % 2 == 0 { 1 } else { 3 }).collect()),
                (0..N).map(|l| if l % 2 == 0 { 1 } else { 3 }).collect(),
            ),
        ] {
            let mut kv = keys();
            set(&mut kv, "attention.head_count_kv", value);
            let mut t = tensors();
            for (l, &count) in heads.iter().enumerate() {
                for stem in ["attn_k", "attn_v"] {
                    let name = format!("blk.{l}.{stem}.weight");
                    t.iter_mut().find(|(n, _)| *n == name).unwrap().1 =
                        vec![E, u64::from(count) * D];
                }
            }
            let read = load(kv, &t, "llama").expect("scalar or per-layer key-head counts");
            for (l, layer) in read.spec.layers.iter().enumerate() {
                let Mixer::Gqa(g) = &layer.mixer else {
                    panic!("layer {l} is not GQA");
                };
                assert_eq!(g.kv_heads, heads[l], "layer {l}");
            }
        }
    }

    #[test]
    fn a_missing_header_key_is_named() {
        for (key, _) in keys() {
            if key == "rope.dimension_count" {
                continue;
            }
            let kv = keys().into_iter().filter(|(k, _)| *k != key).collect();
            let error = load(kv, &tensors(), "llama").expect_err(key);
            assert!(error.contains(key), "{key}: {error}");
        }
    }

    #[test]
    fn incompatible_metadata_is_named() {
        for (key, value) in [
            ("dense_2_feat_in", V::U32(16)),
            ("dense_3_feat_in", V::U32(64)),
            ("dense_3_feat_out", V::U32(16)),
        ] {
            let mut kv = keys();
            set(&mut kv, key, value);
            let error = load(kv, &tensors(), "llama").expect_err(key);
            assert!(error.contains(key), "{key}: {error}");
        }
        for (key, value) in [
            ("attention.key_length_swa", V::U32(16)),
            ("attention.value_length_swa", V::U32(16)),
            ("rope.dimension_count_swa", V::U32(4)),
            ("attention.key_length_swa", V::Bool(true)),
            ("attention.value_length_swa", V::Bool(true)),
            ("rope.dimension_count_swa", V::Bool(true)),
            (
                "attention.sliding_window_pattern",
                V::Bools(vec![true; N as usize]),
            ),
            ("attention.sliding_window_pattern", V::Bools(vec![true; 23])),
            (
                "attention.sliding_window_pattern",
                V::I32s(vec![1; N as usize]),
            ),
            ("attention.sliding_window_pattern", V::U32(5)),
            ("attention.sliding_window_pattern", V::U32(0)),
            ("attention.sliding_window_pattern", V::Bool(true)),
        ] {
            let mut kv = keys();
            kv.push((key, value));
            let error = load(kv, &tensors(), "llama").expect_err(key);
            assert!(error.contains(key), "{key}: {error}");
        }
    }

    #[test]
    fn both_dense_maps_and_trunk_tensor_shapes_are_required() {
        for name in [
            "dense_2.weight",
            "dense_3.weight",
            "blk.5.attn_q.weight",
            "blk.23.post_ffw_norm.weight",
        ] {
            let mut t = tensors();
            t.retain(|(n, _)| n != name);
            let error = load(keys(), &t, "llama").expect_err(name);
            assert!(error.contains(name), "{error}");
            let mut t = tensors();
            t.iter_mut().find(|(n, _)| n == name).unwrap().1.reverse();
            if t.iter().find(|(n, _)| n == name).unwrap().1.len() == 1 {
                t.iter_mut().find(|(n, _)| n == name).unwrap().1[0] += 1;
            }
            let error = load(keys(), &t, "llama").expect_err(name);
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn optional_output_and_explicit_architecture_facts_are_checked() {
        let mut t = tensors();
        t.push(("output.weight".into(), vec![E, 2]));
        let mut kv = keys();
        kv.extend([
            ("attention.causal", V::Bool(false)),
            (
                "attention.sliding_window_pattern",
                V::Bools((0..N).map(|l| (l + 1) % 6 != 0).collect()),
            ),
            ("rope.freq_base_swa", V::F32(5000.0)),
            ("attention.key_length_swa", V::U32(D as u32)),
            ("attention.value_length_swa", V::U32(D as u32)),
            ("rope.dimension_count_swa", V::U32(D as u32)),
        ]);
        let read = load(kv, &t, "llama").expect("explicit facts and optional output");
        assert_eq!(read.defaults.len(), 1);
        assert!(read.defaults[0].contains("tokenizer.ggml.pre = llama"));
        let Mixer::Gqa(g) = &read.spec.layers[0].mixer else {
            panic!("not GQA");
        };
        assert_eq!(g.rope.base, 5000.0);
        let absent = keys()
            .into_iter()
            .filter(|(k, _)| *k != "rope.dimension_count")
            .collect();
        let inferred =
            load(absent, &tensors(), "llama").expect("head-sized rope without a dimension key");
        let Mixer::Gqa(g) = &inferred.spec.layers[0].mixer else {
            panic!("not GQA");
        };
        assert_eq!(g.rope.dims, D as u32);
        assert!(
            inferred
                .defaults
                .iter()
                .any(|s| s.contains("rope.dimension_count = attention.key_length"))
        );
        t.last_mut().unwrap().1 = vec![2, E];
        assert!(
            load(keys(), &t, "llama")
                .unwrap_err()
                .contains("output.weight")
        );
        let mut t = tensors();
        t.push(("blk.0.inp_gate.weight".into(), vec![E, E]));
        assert!(
            load(keys(), &t, "llama")
                .unwrap_err()
                .contains("blk.0.inp_gate.weight")
        );
        assert!(
            load(keys(), &tensors(), "gemma4")
                .unwrap_err()
                .contains("tokenizer.ggml.model")
        );
    }

    #[test]
    fn a_scalar_six_layer_pattern_matches_the_architecture() {
        let mut kv = keys();
        kv.push(("attention.sliding_window_pattern", V::U32(6)));
        let read = load(kv, &tensors(), "llama").expect("a six-layer scalar pattern");
        let encoder = read.spec.encoder.unwrap();
        assert_eq!(
            encoder.reach,
            (0..N)
                .map(|l| ((l + 1) % 6 != 0).then_some(5))
                .collect::<Vec<_>>()
        );
        assert!(
            read.defaults
                .iter()
                .all(|s| !s.starts_with("attention.sliding_window_pattern"))
        );
    }

    #[test]
    fn tokenizer_pre_rewrites_are_recorded() {
        for pre in [None, Some("default"), Some("llama")] {
            let mut global = vec![("tokenizer.ggml.model", V::Str("llama"))];
            if let Some(pre) = pre {
                global.push(("tokenizer.ggml.pre", V::Str(pre)));
            }
            let read = load_with(keys(), &tensors(), &global).unwrap();
            assert_eq!(read.spec.chat.pre, "llama");
            let rewrite = read
                .defaults
                .iter()
                .find(|s| s.starts_with("tokenizer.ggml.pre"));
            if pre == Some("llama") {
                assert!(rewrite.is_none());
            } else {
                assert_eq!(
                    rewrite.unwrap(),
                    &format!(
                        "tokenizer.ggml.pre = llama (tokenizer.ggml.model llama; the file says {pre:?})"
                    )
                );
            }
        }
        let global = [
            ("tokenizer.ggml.model", V::Str("llama")),
            ("tokenizer.ggml.pre", V::Bool(true)),
        ];
        assert!(
            load_with(keys(), &tensors(), &global)
                .unwrap_err()
                .contains("tokenizer.ggml.pre")
        );
    }
}
