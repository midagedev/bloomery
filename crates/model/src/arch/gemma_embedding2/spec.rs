//! The `gemma-embedding2` reader's [`ModelSpec`]. Every layer is grouped
//! query attention with a per-head q/k norm and NeoX rope of its kind's
//! head, and a GELU GLU of `feed_forward_length`. The [`EncoderSpec`] holds
//! what the trunk does not: the sliding layers' reach, the per-layer input's
//! width and the output projection's, and the parts llama.cpp's graph runs
//! past a decoder's layer (the token rows' scale, the values' norm, the
//! norms after both blocks, the layer's scalar). No tool parser and no
//! reasoning span: the model writes no text.

use gguf::{Split, Value};
use models::{Arch, ChatSpec, EncoderSpec, LayerSpec, ModelSpec};

use super::hparams::Hparams;
use super::roles;
use crate::arch::{Read, spec_u32, template_of};
use crate::placement::PlacementError;

/// `tokenizer.ggml.model`'s value, from which llama.cpp sets the
/// pre-tokenizer to the same name (`src/llama-vocab.cpp`, the `gemma4`
/// tokenizer arm).
const GEMMA4: &str = "gemma4";

/// `split`'s description and roles. The one default recorded is the
/// pre-tokenizer when the file carries no `tokenizer.ggml.pre`.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let mut defaults = Vec::new();
    let chat = chat(split, &mut defaults)?;
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults,
    })
}

/// The file's tokenizer and template. The tokenizer model is `gemma4`; the
/// pre-tokenizer is `gemma4` whether the file names it or not, as llama.cpp
/// sets it. Another name in either key is refused by name.
fn chat(split: &Split, defaults: &mut Vec<String>) -> Result<ChatSpec, PlacementError> {
    let mut named = |key: &str, required: bool| -> Result<(), PlacementError> {
        let detail = match split.value(key).map(Value::as_str) {
            Some(Some(GEMMA4)) => return Ok(()),
            None if !required => {
                defaults.push(format!(
                    "{key} = {GEMMA4} (tokenizer.ggml.model {GEMMA4}, as llama.cpp sets it)"
                ));
                return Ok(());
            }
            None => "is absent".to_string(),
            Some(None) => "is not a string".to_string(),
            Some(Some(other)) => format!("is {other:?}, not {GEMMA4}"),
        };
        Err(PlacementError::Metadata {
            key: key.to_string(),
            detail,
        })
    };
    named("tokenizer.ggml.model", true)?;
    named("tokenizer.ggml.pre", false)?;
    Ok(ChatSpec {
        pre: GEMMA4.to_string(),
        template: template_of(split)?,
        tools: None,
        reasoning: None,
    })
}

/// The description of the file `hp` was read from. A per-layer list of
/// another length than `n_layer` is an error naming its key.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let reach = crate::arch::gemma_encoder::reach(hp.n_layer, hp.kv.len(), &hp.sliding, hp.window)?;
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let a = hp.attn(l);
        let (head_key, rope_key) = if hp.sliding[l] {
            ("attention.key_length_swa", "rope.dimension_count_swa")
        } else {
            ("attention.key_length", "rope.dimension_count")
        };
        crate::arch::gemma_encoder::layer_spec(hp.n_head, hp.kv[l], a, hp.n_ff, head_key, rope_key)
    };
    Ok(ModelSpec {
        arch: Arch::GemmaEmbedding2,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: Some(EncoderSpec {
            reach,
            per_layer: Some(spec_u32("embedding_length_per_layer_input", hp.per_layer)?),
            proj: vec![spec_u32("embedding_length_out", hp.n_out)?],
            embed_scale: true,
            value_norm: true,
            post_norms: true,
            layer_scale: true,
        }),
        logit_epilogue: None,
        chat,
        loop_: None,
    })
}

#[cfg(test)]
mod tests {
    use models::{Act, Ffn, Mixer, Need, RopeMode};

    use super::super::hparams::{Hparams, key_bounds};
    use super::super::names;
    use crate::arch::coverage;
    use crate::arch::synthetic::V;
    use crate::arch::{Arch, spec};

    /// Six layers, the third and the sixth full.
    const N: usize = 6;
    const SLIDING: [bool; N] = [true, true, false, true, true, false];
    const EMBD: u64 = 32;
    const HEADS: u64 = 4;
    /// Key and value heads of a sliding and of a full layer.
    const KV: [i32; N] = [2, 2, 1, 2, 2, 1];
    const HEAD_SWA: u64 = 8;
    const HEAD: u64 = 16;
    const FF: u64 = 64;
    const PLE: u64 = 16;
    const OUT: u64 = 48;
    const WINDOW: u32 = 11;

    fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(N as u32)),
            ("embedding_length", V::U32(EMBD as u32)),
            ("context_length", V::U32(128)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::I32s(KV.to_vec())),
            ("attention.key_length", V::U32(HEAD as u32)),
            ("attention.value_length", V::U32(HEAD as u32)),
            ("attention.key_length_swa", V::U32(HEAD_SWA as u32)),
            ("attention.value_length_swa", V::U32(HEAD_SWA as u32)),
            ("rope.dimension_count", V::U32(HEAD as u32)),
            ("rope.dimension_count_swa", V::U32(HEAD_SWA as u32)),
            ("rope.freq_base", V::F32(1e6)),
            ("rope.freq_base_swa", V::F32(1e4)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.sliding_window", V::U32(WINDOW)),
            (
                "attention.sliding_window_pattern",
                V::Bools(SLIDING.to_vec()),
            ),
            ("attention.shared_kv_layers", V::U32(0)),
            ("attention.causal", V::Bool(false)),
            ("embedding_length_per_layer_input", V::U32(PLE as u32)),
            ("embedding_length_out", V::U32(OUT as u32)),
            ("pooling_type", V::U32(1)),
        ]
    }

    /// Layer `l`'s tensors at head `d` and `kv` key heads.
    fn layer(l: usize, d: u64, kv: u64, out: &mut Vec<(String, Vec<u64>)>) {
        let (e, q, k) = (EMBD, HEADS * d, kv * d);
        for (stem, dims) in [
            (names::ATTN_NORM, vec![e]),
            (names::ATTN_Q, vec![e, q]),
            (names::ATTN_K, vec![e, k]),
            (names::ATTN_V, vec![e, k]),
            (names::ATTN_OUTPUT, vec![q, e]),
            (names::ATTN_Q_NORM, vec![d]),
            (names::ATTN_K_NORM, vec![d]),
            (names::POST_ATTENTION_NORM, vec![e]),
            (names::FFN_NORM, vec![e]),
            (names::FFN_GATE, vec![e, FF]),
            (names::FFN_UP, vec![e, FF]),
            (names::FFN_DOWN, vec![FF, e]),
            (names::POST_FFW_NORM, vec![e]),
            (names::INP_GATE, vec![e, PLE]),
            (names::PROJ, vec![PLE, e]),
            (names::POST_NORM, vec![e]),
            (names::LAYER_OUTPUT_SCALE, vec![1]),
        ] {
            out.push((format!("blk.{l}.{stem}"), dims));
        }
    }

    fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut out = vec![
            (names::token_embd(), vec![EMBD, 2]),
            (
                names::PER_LAYER_MODEL_PROJ.to_string(),
                vec![EMBD, PLE * N as u64],
            ),
            (names::PER_LAYER_PROJ_NORM.to_string(), vec![PLE]),
            (names::output_norm(), vec![EMBD]),
            (names::output(), vec![EMBD, OUT]),
        ];
        for l in 0..N {
            let d = if SLIDING[l] { HEAD_SWA } else { HEAD };
            layer(l, d, u64::from(KV[l] as u32), &mut out);
        }
        out
    }

    fn load_with(
        tag: &str,
        arch: &str,
        kv: Vec<(&str, V)>,
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<crate::arch::Read, String> {
        let path = crate::arch::synthetic::header_shaped(tag, arch, &kv, global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    const GLOBAL: [(&str, V); 2] = [
        ("tokenizer.ggml.model", V::Str("gemma4")),
        ("tokenizer.chat_template", V::Str("{{ messages }}")),
    ];

    fn load(
        tag: &str,
        kv: Vec<(&str, V)>,
        tensors: &[(String, Vec<u64>)],
    ) -> Result<crate::arch::Read, String> {
        load_with(tag, "gemma-embedding2", kv, &GLOBAL, tensors)
    }

    fn set_key(kv: &mut [(&str, V)], key: &str, value: V) {
        let row = kv.iter_mut().find(|(k, _)| *k == key).expect(key);
        row.1 = value;
    }

    /// The sliding layers take the `_swa` head and rope, the full layers the
    /// plain ones; each sliding layer reaches half the window on each side,
    /// a full one every key. A reader that gives every layer one head, or
    /// rounds an odd window up or reaches the whole window on a side, fails
    /// this. A head or rope that does not fit names its layer kind's key.
    /// Hparams whose pattern is shorter than the layers are refused by
    /// `spec_of`, not indexed.
    #[test]
    fn a_sliding_layer_takes_the_swa_head_and_half_the_window() {
        let read = load("gemma-emb2-ok", keys(), &tensors()).expect("the header reads");
        let s = &read.spec;
        assert_eq!(s.arch, models::Arch::GemmaEmbedding2);
        assert_eq!((s.hidden, s.vocab, s.ctx_train), (EMBD as u32, 2, 128));
        assert_eq!(s.layers.len(), N);
        assert!(s.mtp.is_empty() && s.hc.is_none() && s.engram.is_none());
        for (l, layer) in s.layers.iter().enumerate() {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("layer {l}: {:?}", layer.mixer);
            };
            let (d, base) = if SLIDING[l] {
                (HEAD_SWA, 1e4)
            } else {
                (HEAD, 1e6)
            };
            assert_eq!(
                (g.heads, g.kv_heads, g.head_dim, g.value_dim),
                (4, KV[l] as u32, d as u32, d as u32),
                "layer {l}"
            );
            assert_eq!(
                (g.rope.mode, g.rope.dims, g.rope.base),
                (RopeMode::Neox, d as u32, base)
            );
            assert!(g.qk_norm && g.window.is_none());
            assert!(matches!(
                layer.ffn,
                Ffn::Dense {
                    ff: 64,
                    act: Act::GeGlu
                }
            ));
        }
        let e = s.encoder.as_ref().expect("an encoder");
        let reach: Vec<_> = SLIDING.iter().map(|&s| s.then_some(5)).collect();
        assert_eq!(e.reach, reach);
        assert_eq!((e.per_layer, e.proj.as_slice()), (Some(16), &[48][..]));
        assert_eq!(s.chat.pre, "gemma4");
        assert!(s.chat.template.is_some() && s.chat.tools.is_none());
        assert_eq!(
            read.defaults,
            ["tokenizer.ggml.pre = gemma4 (tokenizer.ggml.model gemma4, as llama.cpp sets it)"]
        );
        for value in [V::U32(2), V::I32s(vec![2; N])] {
            let mut kv = keys();
            set_key(&mut kv, "attention.head_count_kv", value);
            let mut t = tensors();
            for (l, &sliding) in SLIDING.iter().enumerate() {
                let head = if sliding { HEAD_SWA } else { HEAD };
                for stem in [names::ATTN_K, names::ATTN_V] {
                    let name = crate::arch::names::blk(l, stem);
                    t.iter_mut().find(|(n, _)| *n == name).unwrap().1 = vec![EMBD, 2 * head];
                }
            }
            let read = load("gemma-emb2-shared-kv", kv, &t).expect("shared key-head counts");
            for layer in &read.spec.layers {
                let Mixer::Gqa(g) = &layer.mixer else {
                    panic!("not GQA");
                };
                assert_eq!(g.kv_heads, 2);
            }
        }
        let path = crate::arch::synthetic::header_shaped(
            "gemma-emb2-hp",
            "gemma-embedding2",
            &keys(),
            &GLOBAL,
            &tensors(),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).expect("the keys read");
        drop(split);
        std::fs::remove_file(&path).expect("remove the synthetic header");
        for (sliding, rope, key) in [
            (false, false, "attention.key_length"),
            (true, false, "attention.key_length_swa"),
            (false, true, "rope.dimension_count"),
            (true, true, "rope.dimension_count_swa"),
        ] {
            let mut bad = hp.clone();
            bad.sliding.fill(sliding);
            let attn = if sliding { &mut bad.swa } else { &mut bad.full };
            let too_wide = u32::MAX as usize + 1;
            if rope {
                attn.rope_dims = too_wide;
            } else {
                attn.head = too_wide;
            }
            let err = super::spec_of(&bad, s.chat.clone()).expect_err(key);
            match err {
                crate::placement::PlacementError::Metadata {
                    key: actual,
                    detail,
                } => {
                    assert_eq!(actual, key);
                    assert_eq!(detail, format!("{too_wide} does not fit u32"));
                }
                other => panic!("{key}: {other}"),
            }
        }
        for key in [
            "attention.sliding_window_pattern",
            "attention.head_count_kv",
        ] {
            let mut bad = hp.clone();
            if key == "attention.sliding_window_pattern" {
                bad.sliding.pop();
            } else {
                bad.kv.pop();
            }
            let err = super::spec_of(&bad, s.chat.clone()).expect_err(key);
            assert!(
                err.to_string()
                    .contains(&format!("{key}: has 5 values for 6 layers")),
                "{err}"
            );
        }
    }

    /// Every row's key bounds at each small length and reach are the keys
    /// llama.cpp's symmetric mask keeps, `|i - j| <= reach`, and every key
    /// with no reach. A bound one key short or long on either side fails
    /// this.
    #[test]
    fn key_bounds_are_the_symmetric_mask() {
        for n in 1..=12usize {
            for reach in (0..=13usize).map(Some).chain([None]) {
                for t in 0..n {
                    let kept: Vec<usize> = (0..n)
                        .filter(|&j| reach.is_none_or(|r| t.abs_diff(j) <= r))
                        .collect();
                    let (first, end) = key_bounds(reach, t, n);
                    assert_eq!(
                        (first..end).collect::<Vec<_>>(),
                        kept,
                        "{n} positions, reach {reach:?}, row {t}"
                    );
                }
            }
        }
    }

    /// The coverage check holds the file to the encoder program's shapes:
    /// the bidirectional attention of both heads, the GELU GLU, the
    /// per-layer input and the output projection at widths it does not run
    /// are items, and no causal flash, SwiGLU layer, tool parser or missing
    /// program is. The value norm, the post norms, the layer scalar and the
    /// token rows' scale run at any width, so none is an item; a reader that
    /// drops one of the encoder's parts, or a program row that covers a
    /// causal need, fails this.
    #[test]
    fn coverage_names_the_encoder_parts() {
        let read = load("gemma-emb2-cover", keys(), &tensors()).expect("the header reads");
        let encoder = read.spec.encoder.as_ref().expect("an encoder");
        assert!(encoder.embed_scale);
        assert!(encoder.value_norm);
        assert!(encoder.post_norms);
        assert!(encoder.layer_scale);
        let needs = models::needs(&read.spec);
        assert!(needs.contains(&(Need::EmbedScale, None)));
        for (l, layer) in read.spec.layers.iter().enumerate() {
            let Mixer::Gqa(gqa) = &layer.mixer else {
                panic!("layer {l}: not GQA");
            };
            for need in [
                Need::ValueNorm { head: gqa.head_dim },
                Need::PostNorms,
                Need::LayerScale,
            ] {
                assert!(needs.contains(&(need.clone(), Some(l as u32))), "{need:?}");
            }
        }
        let items: Vec<String> = coverage::check(&read.spec, &read.tensors)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        for want in [
            "bidirectional GQA flash, head 8, group 2, 5 keys each side",
            "bidirectional GQA flash, head 16, group 4",
            "dense GELU GLU layer: ff 64",
            "a per-layer input of 16 values from the token rows",
            "the output projections [48], then the mean of the rows",
        ] {
            assert!(items.iter().any(|i| i == want), "{want}: {items:?}");
        }
        for not in [
            "GQA flash, head",
            "dense SwiGLU",
            "tool-call parser",
            "a layer program",
            "a gainless",
            "RMS norms on",
            "a trained scalar",
            "token rows scaled",
        ] {
            assert!(
                !items.iter().any(|i| i.starts_with(not)),
                "{not}: {items:?}"
            );
        }
    }

    /// A full layer whose tensors are shaped at the sliding head is refused
    /// by the tensor's name: the head is each layer's kind's. A missing
    /// tensor is named before any of other dims.
    #[test]
    fn a_tensor_missing_or_of_other_dims_is_named() {
        let mut t = tensors();
        t.retain(|(n, _)| !n.starts_with("blk.2."));
        layer(2, HEAD_SWA, 1, &mut t);
        let err = load("gemma-emb2-head", keys(), &t).expect_err("layer 2 at head 8");
        assert!(
            err.contains("blk.2.attn_q.weight") && err.contains("want [32, 64]"),
            "{err}"
        );
        t.retain(|(n, _)| n != "blk.4.ffn_up.weight");
        t.retain(|(n, _)| n != "blk.5.ffn_down.weight");
        let err = load("gemma-emb2-gone", keys(), &t).expect_err("no blk.4.ffn_up");
        assert!(
            err.contains("tensor blk.4.ffn_up.weight: is not in the file")
                && err.contains("2 tensor(s) the model needs are missing")
                && err.contains("blk.4.ffn_up.weight, blk.5.ffn_down.weight")
                && !err.contains("blk.2."),
            "{err}"
        );
    }

    /// Each key the reader takes is required by name; llama.cpp's defaults
    /// are not taken.
    #[test]
    fn a_missing_key_is_named() {
        for (key, _) in keys() {
            if matches!(key, "attention.causal" | "attention.shared_kv_layers") {
                continue;
            }
            let kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
            let err = load("gemma-emb2-key", kv, &tensors()).expect_err(key);
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    /// A file that pools otherwise, attends causally, shares key and value
    /// layers, has a value head other than its key head, a layer with no key
    /// head or with key heads that do not divide the query heads, an odd rope
    /// count or one larger than the head, a nonpositive rope base, or a
    /// pattern of another length or item type is refused by the key's name.
    #[test]
    fn a_key_of_another_model_is_named() {
        for key in [
            "rope.freq_base",
            "rope.freq_base_swa",
            "attention.layer_norm_rms_epsilon",
        ] {
            for value in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut kv = keys();
                set_key(&mut kv, key, V::F32(value));
                let err = load("gemma-emb2-float", kv, &tensors()).expect_err(key);
                assert!(err.contains(key), "{key}: {err}");
            }
        }
        for (key, value, want) in [
            ("pooling_type", V::U32(2), "is 2"),
            ("attention.causal", V::Bool(true), "is true"),
            ("attention.shared_kv_layers", V::U32(4), "is 4"),
            ("attention.value_length_swa", V::U32(16), "is 16"),
            (
                "attention.head_count_kv",
                V::I32s(vec![2, 2, 0, 2, 2, 1]),
                "is 0 at layer 2",
            ),
            (
                "attention.head_count_kv",
                V::I32s(vec![2, 2, 3, 2, 2, 1]),
                "is 3 at layer 2",
            ),
            (
                "rope.dimension_count_swa",
                V::U32(7),
                "is 7, for a head of 8",
            ),
            (
                "rope.dimension_count_swa",
                V::U32(10),
                "is 10, for a head of 8",
            ),
            ("rope.freq_base", V::F32(0.0), "is 0"),
            (
                "attention.sliding_window_pattern",
                V::Bools(vec![true; N - 1]),
                "has 5 values",
            ),
            (
                "attention.sliding_window_pattern",
                V::I32s(vec![1; N]),
                "has no bool at layer 0",
            ),
        ] {
            let mut kv = keys();
            set_key(&mut kv, key, value);
            let err = load("gemma-emb2-other", kv, &tensors()).expect_err(key);
            assert!(err.contains(key) && err.contains(want), "{key}: {err}");
        }
        for value in [V::Bool(true), V::Bools(vec![true; N + 1])] {
            let key = "attention.sliding_window_pattern";
            let mut kv = keys();
            set_key(&mut kv, key, value);
            let err = load("gemma-emb2-pattern", kv, &tensors()).expect_err(key);
            assert!(err.contains(key), "{err}");
        }
    }

    /// A tokenizer other than `gemma4`, or a pre-tokenizer of another name,
    /// is refused by name; the file's own `gemma4` is no default.
    #[test]
    fn a_tokenizer_other_than_gemma4_is_named() {
        let llama = [("tokenizer.ggml.model", V::Str("llama"))];
        let err = load_with(
            "gemma-emb2-tok",
            "gemma-embedding2",
            keys(),
            &llama,
            &tensors(),
        )
        .expect_err("llama tokenizer");
        assert!(err.contains("tokenizer.ggml.model"), "{err}");
        let pre = [
            ("tokenizer.ggml.model", V::Str("gemma4")),
            ("tokenizer.ggml.pre", V::Str("qwen2")),
        ];
        let err = load_with(
            "gemma-emb2-pre",
            "gemma-embedding2",
            keys(),
            &pre,
            &tensors(),
        )
        .expect_err("qwen2 pre");
        assert!(err.contains("tokenizer.ggml.pre"), "{err}");
        let own = [
            ("tokenizer.ggml.model", V::Str("gemma4")),
            ("tokenizer.ggml.pre", V::Str("gemma4")),
        ];
        let read = load_with(
            "gemma-emb2-own",
            "gemma-embedding2",
            keys(),
            &own,
            &tensors(),
        )
        .expect("gemma4 pre");
        assert!(read.defaults.is_empty(), "{:?}", read.defaults);
        assert!(read.spec.chat.template.is_none());
    }

    /// A tensor no table holds is refused by name.
    #[test]
    fn an_unknown_tensor_is_named() {
        let mut t = tensors();
        t.push(("per_layer_token_embd.weight".to_string(), vec![1]));
        let err = load("gemma-emb2-extra", keys(), &t).expect_err("extra");
        assert!(err.contains("per_layer_token_embd.weight"), "{err}");
    }

    /// PIN(2026-10-15): gemma-embedding now has a reader; its header tests pin acceptance.
    #[test]
    fn other_gemma_embedding_strings_are_unknown() {
        for arch in ["gemma_embedding", "gemma_embedding2", "gemma-embedding3"] {
            assert!(Arch::from_name(arch).is_err(), "{arch} is known");
            let err = load_with(arch, arch, vec![], &[], &[]).expect_err(arch);
            assert_eq!(err, format!("unsupported architecture \"{arch}\""));
        }
        assert_eq!(
            Arch::from_name("gemma-embedding2").unwrap(),
            Arch::GemmaEmbedding2
        );
    }
}
