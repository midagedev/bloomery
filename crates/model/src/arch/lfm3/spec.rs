//! The `lfm3` reader's [`ModelSpec`]. A short-conv layer carries a [`Mixer::ShortConv`]. An
//! attention layer is grouped-query with per-head query and key norms, the NeoX rope of its own θ
//! over the whole head, and a sigmoid output gate of one value per head ([`OutGate::ProjHead`]);
//! a sliding layer attends the last `attention.sliding_window` positions and a full layer every
//! one, and only a full layer's rope divides by the file's `rope_freqs.weight`. Every layer's
//! feed-forward block is a dense SwiGLU, and both sublayers join the stream through the sandwich
//! norms ([`Residual::Sandwich`]). No next-token layer is read. The tool calls are the
//! `<tool_call>` parser and the `<think>` span is read.

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, OutGate, ReasoningFormat, Residual,
    Rope, RopeMode, ShortConv, ToolFormat,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// The tool-call markup the family's template teaches: `<tool_call>` with `<function=…>` and
/// `<parameter=…>`.
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::QwenXml);

/// `split`'s description and roles. This reader takes no key default.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(ReasoningFormat::THINK))?;
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: Vec::new(),
    })
}

/// The description of the file `hp` was read from.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = match hp.kinds[l] {
            Kind::Conv => Mixer::ShortConv(ShortConv {
                taps: spec_u32("shortconv.l_cache", hp.taps)?,
            }),
            kind => Mixer::Gqa(gqa(hp, l, kind)?),
        };
        Ok(LayerSpec {
            mixer,
            ffn: Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.ff[l])?,
                act: Act::SwiGlu { limit: None },
            },
            residual: Residual::Sandwich {
                eps: hp.rms_eps,
                add: [1.0, 1.0],
            },
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::Lfm3,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: None,
        chat,
    })
}

/// Attention layer `l` of `kind`.
fn gqa(hp: &Hparams, l: usize, kind: Kind) -> Result<Gqa, PlacementError> {
    let sliding = kind == Kind::Sliding;
    Ok(Gqa {
        heads: spec_u32("attention.head_count", hp.n_head)?,
        kv_heads: spec_u32("attention.head_count_kv", hp.kv[l])?,
        head_dim: spec_u32("attention.key_length", hp.key_len)?,
        value_dim: spec_u32("attention.value_length", hp.value_len)?,
        rope: Rope {
            mode: RopeMode::Neox,
            dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
            base: if sliding {
                hp.rope_base_swa
            } else {
                hp.rope_base
            },
            yarn: None,
            factors: kind == Kind::Full && hp.rope_factors,
        },
        qk_norm: true,
        o_norm: false,
        out_gate: OutGate::ProjHead,
        select: None,
        window: if sliding {
            Some(spec_u32("attention.sliding_window", hp.window)?)
        } else {
            None
        },
        sinks: false,
        value_scale: None,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use models::{Act, Ffn, Mixer, Residual, RopeMode};

    use super::super::names::{
        ATTN_GATE, ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_V,
        CONV_IN, CONV_OUT, CONV_TAPS, FFN_DOWN, FFN_GATE, FFN_NORM, FFN_UP, POST_ATTN_NORM,
        POST_FFN_NORM, ROPE_FREQS, blk, output, output_norm, token_embd,
    };
    use crate::arch::synthetic::{Ty, V, header_shaped, header_typed};
    use crate::arch::{Read, coverage};
    use crate::placement::Role;

    /// Six layers: short conv, short conv, full, short conv, sliding, short conv.
    pub(in crate::arch::lfm3) const KINDS: &str = "ccfcsc";
    pub(in crate::arch::lfm3) const N: usize = 6;
    pub(in crate::arch::lfm3) const EMBD: u64 = 32;
    pub(in crate::arch::lfm3) const HEADS: u64 = 4;
    pub(in crate::arch::lfm3) const KV: u32 = 2;
    pub(in crate::arch::lfm3) const HEAD: u64 = 8;
    const FF: u64 = 64;
    pub(in crate::arch::lfm3) const TAPS: u64 = 7;
    pub(in crate::arch::lfm3) const WINDOW: u32 = 16;

    /// The header's keys, the shapes of the real file's at a size a test can write.
    pub(in crate::arch::lfm3) fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(N as u32)),
            ("embedding_length", V::U32(EMBD as u32)),
            ("context_length", V::U32(128)),
            ("vocab_size", V::U32(EMBD as u32)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::I32s(vec![0, 0, 2, 0, 2, 0])),
            ("attention.key_length", V::U32(HEAD as u32)),
            ("attention.value_length", V::U32(HEAD as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.sliding_window", V::U32(WINDOW)),
            (
                "attention.sliding_window_pattern",
                V::Bools(vec![false, false, false, false, true, false]),
            ),
            ("rope.freq_base", V::F32(1e6)),
            ("rope.freq_base_swa", V::F32(1e4)),
            ("rope.dimension_count", V::U32(HEAD as u32)),
            ("shortconv.l_cache", V::U32(TAPS as u32)),
        ]
    }

    /// Layer `l`'s tensors for `kind` (`c` short conv, `f` full or `s` sliding attention).
    fn layer(l: usize, kind: char, out: &mut Vec<(String, Vec<u64>)>) {
        let mut one = |stem: &str, dims: Vec<u64>| out.push((blk(l, stem), dims));
        let e = EMBD;
        one(ATTN_NORM, vec![e]);
        one(POST_ATTN_NORM, vec![e]);
        one(FFN_NORM, vec![e]);
        one(POST_FFN_NORM, vec![e]);
        one(FFN_GATE, vec![e, FF]);
        one(FFN_UP, vec![e, FF]);
        one(FFN_DOWN, vec![FF, e]);
        match kind {
            'c' => {
                one(CONV_IN, vec![e, 3 * e]);
                one(CONV_TAPS, vec![TAPS, e]);
                one(CONV_OUT, vec![e, e]);
            }
            'f' | 's' => {
                let (h, kv, d) = (HEADS, u64::from(KV), HEAD);
                one(ATTN_Q, vec![e, h * d]);
                one(ATTN_K, vec![e, kv * d]);
                one(ATTN_V, vec![e, kv * d]);
                one(ATTN_OUTPUT, vec![h * d, e]);
                one(ATTN_GATE, vec![e, h]);
                one(ATTN_Q_NORM, vec![d]);
                one(ATTN_K_NORM, vec![d]);
            }
            _ => panic!("kind {kind}"),
        }
    }

    /// The file's tensors: the tied head (no `output.weight`), the rope's factors, the layers.
    pub(in crate::arch::lfm3) fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut out = vec![
            (token_embd(), vec![EMBD, EMBD]),
            (output_norm(), vec![EMBD]),
            (ROPE_FREQS.to_string(), vec![HEAD / 2]),
        ];
        for (l, kind) in KINDS.chars().enumerate() {
            layer(l, kind, &mut out);
        }
        out
    }

    /// The global keys the real file carries that the reader takes: the pre-tokenizer.
    fn global() -> [(&'static str, V); 1] {
        [("tokenizer.ggml.pre", V::Str("lfm2"))]
    }

    /// What the registry's reader makes of the header with `kv` and `tensors`: its value, or its
    /// error as text.
    pub(in crate::arch::lfm3) fn load(
        tag: &str,
        kv: Vec<(&str, V)>,
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Read, String> {
        let path = header_shaped(tag, "lfm3", &kv, &global(), tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    fn ok(tag: &str) -> Read {
        load(tag, keys(), &tensors()).expect("the header reads")
    }

    fn set_key(kv: &mut [(&str, V)], key: &str, value: V) {
        let row = kv.iter_mut().find(|(k, _)| *k == key).expect(key);
        row.1 = value;
    }

    /// Layers 0, 1, 3 and 5 are short convs of seven taps, layer 2 is full attention and layer 4
    /// sliding, each over a dense SwiGLU block and the sandwich residual. A short-conv layer read
    /// as attention, or a sliding layer as full, fails this.
    #[test]
    fn the_layers_read_as_their_kinds() {
        let read = ok("lfm3-ok");
        let s = &read.spec;
        assert_eq!(s.arch, models::Arch::Lfm3);
        assert_eq!(
            (s.hidden, s.vocab, s.ctx_train),
            (EMBD as u32, EMBD as u32, 128)
        );
        assert_eq!(s.rms_eps.to_bits(), 0x3727_c5ac);
        assert!(s.mtp.is_empty() && s.hc.is_none() && s.engram.is_none() && s.encoder.is_none());
        assert!(s.logit_epilogue.is_none());
        assert!(read.defaults.is_empty());
        assert_eq!(s.chat.pre, "lfm2");
        assert_eq!(s.chat.tools, Some(models::ToolFormat::QwenXml));
        assert_eq!(s.chat.reasoning, Some(models::ReasoningFormat::THINK));
        assert_eq!(s.layers.len(), N);
        let kind = |l: usize| {
            let layer = &s.layers[l];
            assert!(matches!(
                layer.ffn,
                Ffn::Dense {
                    ff: 64,
                    act: Act::SwiGlu { limit: None }
                }
            ));
            assert_eq!(
                layer.residual,
                Residual::Sandwich {
                    eps: s.rms_eps,
                    add: [1.0, 1.0]
                }
            );
            assert!(layer.extras.is_empty());
            match &layer.mixer {
                Mixer::ShortConv(c) => {
                    assert_eq!(c.taps, 7);
                    'c'
                }
                Mixer::Gqa(g) => {
                    assert_eq!((g.heads, g.kv_heads, g.head_dim, g.value_dim), (4, 2, 8, 8));
                    assert!(g.qk_norm && !g.o_norm && !g.sinks);
                    assert!(g.select.is_none() && g.value_scale.is_none());
                    assert_eq!(g.out_gate, models::OutGate::ProjHead);
                    assert_eq!((g.rope.mode, g.rope.dims), (RopeMode::Neox, 8));
                    assert!(g.rope.yarn.is_none());
                    match g.window {
                        None => {
                            assert_eq!(g.rope.base, 1e6);
                            assert!(g.rope.factors);
                            'f'
                        }
                        Some(w) => {
                            assert_eq!(w, WINDOW);
                            assert_eq!(g.rope.base, 1e4);
                            assert!(!g.rope.factors);
                            's'
                        }
                    }
                }
                other => panic!("layer {l}: {other:?}"),
            }
        };
        assert_eq!((0..N).map(kind).collect::<String>(), KINDS);
    }

    /// The rope's factors are the file's own tensor, on the full layers only: a file without the
    /// table reads with no factors on any layer, and the description then needs none.
    #[test]
    fn the_factors_are_the_files_table() {
        let mut bare = tensors();
        bare.retain(|(n, _)| n != ROPE_FREQS);
        let read = load("lfm3-no-factors", keys(), &bare).expect("a file with no factor table");
        for layer in &read.spec.layers {
            if let Mixer::Gqa(g) = &layer.mixer {
                assert!(!g.rope.factors);
            }
        }
        assert!(
            !models::needs(&read.spec)
                .iter()
                .any(|(n, _)| matches!(n, models::Need::RopeFactors))
        );
    }

    /// Every tensor has a role: the rope's factors are read once by the load (`Unread`, on no
    /// layer), the token embedding is gathered, the norm is the head's, and a layer's tensors sit on
    /// the layer. A role table that drops the factors, or sends a post-norm to the head, fails this.
    #[test]
    fn every_tensor_has_its_role() {
        let read = ok("lfm3-roles");
        let role = |name: &str| {
            let t = read
                .tensors
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name}: not classified"));
            (t.role, t.layer)
        };
        assert_eq!(role(ROPE_FREQS), (Role::Unread, None));
        assert_eq!(role("token_embd.weight"), (Role::TokenEmbedding, None));
        assert_eq!(role("output_norm.weight"), (Role::Head, None));
        assert_eq!(role("blk.2.attn_gate.weight"), (Role::Attention, Some(2)));
        assert_eq!(
            role("blk.0.shortconv.conv.weight"),
            (Role::Attention, Some(0))
        );
        assert_eq!(
            role("blk.3.post_attention_norm.weight"),
            (Role::Attention, Some(3))
        );
        assert_eq!(role("blk.3.post_ffw_norm.weight"), (Role::FfnNorm, Some(3)));
        assert_eq!(role("blk.4.ffn_down.weight"), (Role::DenseFfn, Some(4)));
        assert_eq!(read.tensors.layers, N);
        assert_eq!((read.tensors.experts, read.tensors.experts_used), (0, 0));
        assert_eq!(read.tensors.tensors.len(), tensors().len());
    }

    /// No program runs the description yet, so the coverage check lists each new part by name and
    /// layer, and the file's pre-tokenizer is one the tokenizer runs. A part a program silently ran
    /// as another would be missing here; a pre-tokenizer alias removed from the tokenizer lists it.
    #[test]
    fn the_coverage_names_every_part_no_body_runs() {
        let read = ok("lfm3-coverage");
        let items = coverage::check(&read.spec, &read.tensors);
        let has = |layer: Option<usize>, feature: &str| {
            items
                .iter()
                .any(|u| u.layer == layer && u.feature == feature)
        };
        for l in [0, 1, 3, 5] {
            assert!(
                has(Some(l), "gated short-conv mixer: 7 taps"),
                "{l}: {items:?}"
            );
        }
        for l in [2, 4] {
            assert!(
                has(
                    Some(l),
                    "attention output gate: sigmoid of its own projection, one gate per head"
                ),
                "{l}: {items:?}"
            );
        }
        assert!(has(Some(2), "rope frequency factors (rope_freqs.weight)"));
        assert!(!has(Some(4), "rope frequency factors (rope_freqs.weight)"));
        for l in 0..N {
            assert!(
                has(
                    Some(l),
                    "sandwich norms: each sublayer's output weight-normed at 0.00001, then added scaled by [1.0, 1.0]"
                ),
                "{l}: {items:?}"
            );
        }
        assert!(has(None, "a layer program for lfm3"), "{items:?}");
        assert!(
            !items.iter().any(|u| u.feature.starts_with("pre-tokenizer")),
            "{items:?}"
        );
    }

    /// The head is the token embedding unless the file carries `output.weight`: a file with one
    /// reads, and its dims are held to the keys'.
    #[test]
    fn an_untied_head_is_read() {
        let mut with = tensors();
        with.push((output(), vec![EMBD, EMBD]));
        let read = load("lfm3-untied", keys(), &with).expect("an untied head reads");
        assert!(
            read.tensors
                .tensors
                .iter()
                .any(|t| t.name == "output.weight" && t.role == Role::Head)
        );
        with.pop();
        with.push((output(), vec![EMBD, EMBD + 1]));
        let err = load("lfm3-untied-dims", keys(), &with).expect_err("an output of other dims");
        assert!(
            err.contains("output.weight") && err.contains("has other dims than the keys give"),
            "{err}"
        );
    }

    /// `lfm3` is read through the registry's row and no `Arch` variant of this crate stands for
    /// it.
    #[test]
    fn the_registry_names_the_reader() {
        assert!(crate::arch::Arch::from_name("lfm3").is_err());
        let row = crate::arch::registry::row_of("lfm3").expect("a row");
        assert!(row.read.is_some() && row.draft.is_none() && row.fixture.is_none());
    }

    /// Each key the reader takes is required by name, but the vocabulary's size, which the token
    /// list stands in for. Dropping the check for one of them lets this header read.
    #[test]
    fn a_missing_key_is_named() {
        let tensors = tensors();
        for (key, _) in keys().into_iter().filter(|(k, _)| *k != "vocab_size") {
            let kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
            let err = load("lfm3-key", kv, &tensors).expect_err(key);
            assert!(
                err.contains(&format!("lfm3.{key}")) && err.contains("absent"),
                "{key}: {err}"
            );
        }
    }

    /// A short-conv width the file does not hold is refused by name: a key of no taps, and a
    /// filter tensor whose dims are not `[taps, channels]`, or whose taps are another width.
    /// Mutant: the check on `l_cache` removed (a conv ring of no row), or the dims read as
    /// `[channels, taps]`.
    #[test]
    fn a_wrong_conv_width_is_named() {
        let mut kv = keys();
        set_key(&mut kv, "shortconv.l_cache", V::U32(0));
        let err = load("lfm3-taps-zero", kv, &tensors()).expect_err("no taps");
        assert!(
            err.contains("lfm3.shortconv.l_cache") && err.contains("is 0"),
            "{err}"
        );
        let name = blk(0, CONV_TAPS);
        for dims in [vec![TAPS - 1, EMBD], vec![EMBD, TAPS]] {
            let mut bad = tensors();
            bad.iter_mut()
                .find(|(n, _)| *n == name)
                .expect("the filters")
                .1 = dims.clone();
            let err = load("lfm3-taps", keys(), &bad).expect_err("another conv width");
            assert!(
                err.contains(&name) && err.contains("has other dims than the keys give the tensor"),
                "{dims:?}: {err}"
            );
        }
    }

    /// A layer pattern the file cannot hold is refused by name: an array of the wrong length for
    /// either per-layer key, a sliding entry on a short-conv layer, an array that makes a layer
    /// another kind than its tensors do, and a key-head count that does not divide the query
    /// heads. Mutant: the sliding entry of a short-conv layer ignored.
    #[test]
    fn a_wrong_layer_pattern_is_named() {
        let tensors = tensors();
        let mut kv = keys();
        set_key(
            &mut kv,
            "attention.head_count_kv",
            V::I32s(vec![0, 0, 2, 0, 2]),
        );
        let err = load("lfm3-kv-len", kv, &tensors).expect_err("short kv array");
        assert!(
            err.contains("lfm3.attention.head_count_kv")
                && err.contains("has 5 values for 6 layers"),
            "{err}"
        );
        let mut kv = keys();
        set_key(
            &mut kv,
            "attention.sliding_window_pattern",
            V::Bools(vec![false, false, false, false, true]),
        );
        let err = load("lfm3-pattern-len", kv, &tensors).expect_err("short pattern");
        assert!(
            err.contains("lfm3.attention.sliding_window_pattern")
                && err.contains("has 5 values for 6 layers"),
            "{err}"
        );
        let mut kv = keys();
        set_key(
            &mut kv,
            "attention.sliding_window_pattern",
            V::Bools(vec![false, false, false, true, true, false]),
        );
        let err = load("lfm3-pattern-conv", kv, &tensors).expect_err("sliding on a short conv");
        assert!(
            err.contains("lfm3.attention.sliding_window_pattern")
                && err.contains("is true at layer 3")
                && err.contains("short-conv"),
            "{err}"
        );
        let mut kv = keys();
        set_key(
            &mut kv,
            "attention.head_count_kv",
            V::I32s(vec![0, 0, 0, 0, 2, 0]),
        );
        let err = load("lfm3-kv-kind", kv, &tensors).expect_err("kv says short conv");
        assert!(
            err.contains("lfm3.attention.head_count_kv")
                && err.contains("is 0 at layer 2")
                && err.contains("Full"),
            "{err}"
        );
        let mut kv = keys();
        set_key(
            &mut kv,
            "attention.head_count_kv",
            V::I32s(vec![0, 0, 2, 0, 3, 0]),
        );
        let err = load("lfm3-kv-divisor", kv, &tensors).expect_err("kv of 3 for 4 heads");
        assert!(
            err.contains("lfm3.attention.head_count_kv")
                && err.contains("is 3 at layer 4 for 4 query heads"),
            "{err}"
        );
    }

    /// A layer that carries the other kind's marker tensor, both, or neither is refused naming
    /// the layer and both markers.
    #[test]
    fn a_layer_with_both_or_neither_marker_is_named() {
        let mut both = tensors();
        both.push((blk(0, ATTN_Q), vec![EMBD, HEADS * HEAD]));
        let err = load("lfm3-both", keys(), &both).expect_err("both markers");
        assert!(
            err.contains("layer 0 needs exactly one of") && err.contains(CONV_IN),
            "{err}"
        );
        let mut neither = tensors();
        neither.retain(|(n, _)| *n != blk(0, CONV_IN));
        let err = load("lfm3-neither", keys(), &neither).expect_err("no marker");
        assert!(err.contains("layer 0 needs exactly one of"), "{err}");
    }

    /// The rope's factor table is one f32 for each pair of the rotated values: a table of another
    /// length, and one of another type, are refused by name. Mutant: the length check on the key
    /// head's width (twice the pairs).
    #[test]
    fn a_rope_table_of_the_wrong_length_or_type_is_named() {
        for dims in [vec![HEAD], vec![HEAD / 2 - 1], vec![1, HEAD / 2]] {
            let mut bad = tensors();
            bad.iter_mut()
                .find(|(n, _)| n == ROPE_FREQS)
                .expect("the table")
                .1 = dims.clone();
            let err = load("lfm3-rope-len", keys(), &bad).expect_err("a table of other dims");
            assert!(
                err.contains(ROPE_FREQS) && err.contains("one factor each"),
                "{dims:?}: {err}"
            );
        }
        let typed: Vec<_> = tensors()
            .into_iter()
            .map(|(n, d)| {
                let ty = if n == ROPE_FREQS { Ty::F16 } else { Ty::F32 };
                (n, d, ty)
            })
            .collect();
        let path = header_typed("lfm3-rope-f16", "lfm3", &keys(), &global(), &typed);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = crate::arch::spec(&split)
            .expect_err("an f16 table")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains(ROPE_FREQS) && err.contains("the rope's factors are f32"),
            "{err}"
        );
    }

    /// `rope.dimension_count` is held to the head: odd, zero and past the key length are refused.
    #[test]
    fn a_rope_wider_than_the_head_is_named() {
        for dims in [0, 7, 16] {
            let mut kv = keys();
            set_key(&mut kv, "rope.dimension_count", V::U32(dims));
            let err = load("lfm3-rope-dims", kv, &tensors()).expect_err("a bad rope width");
            assert!(err.contains("lfm3.rope.dimension_count"), "{dims}: {err}");
        }
    }

    /// A scaled rope is refused by name: this reader runs the plain rope.
    #[test]
    fn a_scaled_rope_is_named() {
        let mut kv = keys();
        kv.push(("rope.scaling.type", V::Str("yarn")));
        let err = load("lfm3-yarn", kv, &tensors()).expect_err("yarn");
        assert!(err.contains("lfm3.rope.scaling.type"), "{err}");
    }

    /// A tensor a layer needs and the file lacks is refused by name, with the count of those
    /// missing: a layer's post-norm, the model's norm, the token embedding.
    #[test]
    fn a_missing_role_is_named() {
        for name in [
            "blk.3.post_ffw_norm.weight",
            "blk.4.attn_gate.weight",
            "blk.1.shortconv.out_proj.weight",
            "output_norm.weight",
            "token_embd.weight",
        ] {
            let mut bad = tensors();
            bad.retain(|(n, _)| n != name);
            let err = load("lfm3-missing", keys(), &bad).expect_err(name);
            assert!(
                err.contains(name) && err.contains("is not in the file"),
                "{name}: {err}"
            );
        }
    }

    /// A tensor no role claims is refused by name, whichever layer it is on: a conv bias, an
    /// attention tensor on a short-conv layer (the layer's markers say short conv), and an
    /// unknown model-level name.
    #[test]
    fn an_unclaimed_tensor_is_named() {
        for name in [
            "blk.0.shortconv.conv.bias",
            "blk.0.attn_k.weight",
            "blk.2.shortconv.conv.weight",
            "rope_freqs.bias",
            "blk.6.attn_norm.weight",
        ] {
            let mut bad = tensors();
            bad.push((name.to_string(), vec![1]));
            let err = load("lfm3-extra", keys(), &bad).expect_err(name);
            assert!(err.contains(name), "{name}: {err}");
        }
    }

    /// The feed-forward length is a count above 0 on every layer.
    #[test]
    fn a_feed_forward_block_of_no_width_is_named() {
        let mut kv = keys();
        set_key(
            &mut kv,
            "feed_forward_length",
            V::I32s(vec![64, 64, 0, 64, 64, 64]),
        );
        let err = load("lfm3-ff", kv, &tensors()).expect_err("no feed-forward width");
        assert!(
            err.contains("lfm3.feed_forward_length") && err.contains("is 0 at layer 2"),
            "{err}"
        );
    }
}
