//! The `nemotron_h` reader's [`ModelSpec`]. A Mamba-2 layer carries a
//! [`Mixer::Mamba2`] and no feed-forward block. An attention layer is grouped
//! query with no rope. An MLP layer is a squared ReLU of that layer's
//! `feed_forward_length` and has no token mixer. No next-token layer is
//! read. The tool calls are the `<tool_call>` parser and the `<think>` span
//! is read.

use gguf::Split;
use models::{
    Act, Arch, Ffn, Gqa, LayerSpec, Mamba2, Mixer, ModelSpec, ReasoningFormat, Residual, Rope,
    RopeMode, ToolFormat,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// The tool-call markup the family's template teaches: `<tool_call>` with
/// `<function=…>` and `<parameter=…>`.
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::QwenXml);

/// `split`'s description and roles. This reader takes no key default.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(ReasoningFormat::ThinkSpan))?;
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` was read from.
pub fn spec_of(hp: &Hparams, chat: models::ChatSpec) -> Result<ModelSpec, PlacementError> {
    let mamba = Mamba2 {
        d_inner: spec_u32("ssm.inner_size", hp.ssm.d_inner)?,
        d_state: spec_u32("ssm.state_size", hp.ssm.d_state)?,
        n_heads: spec_u32("ssm.time_step_rank", hp.ssm.n_heads)?,
        head_dim: spec_u32("ssm.inner_size / ssm.time_step_rank", hp.ssm.head_dim)?,
        n_groups: spec_u32("ssm.group_count", hp.ssm.n_groups)?,
        d_conv: spec_u32("ssm.conv_kernel", hp.ssm.d_conv)?,
        d_in_proj: spec_u32("ssm input projection", hp.ssm.d_in_proj)?,
    };
    let no_block = Ffn::Dense {
        ff: 0,
        act: Act::ReluSqr,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let (mixer, ffn) = match hp.kinds[l] {
            Kind::Mamba => (Mixer::Mamba2(mamba), no_block.clone()),
            Kind::Attention => (
                Mixer::Gqa(Gqa {
                    heads: spec_u32("attention.head_count", hp.n_head)?,
                    kv_heads: spec_u32("attention.head_count_kv", hp.kv[l])?,
                    head_dim: spec_u32("attention.key_length", hp.key_len)?,
                    value_dim: spec_u32("attention.value_length", hp.value_len)?,
                    // No value is rotated. `rope.dimension_count` was read
                    // and is not this layer's.
                    rope: Rope {
                        mode: RopeMode::None,
                        dims: 0,
                        base: 0.0,
                        yarn: None,
                    },
                    qk_norm: false,
                    out_gate: false,
                    select: None,
                    window: None,
                    sinks: false,
                    value_scale: None,
                }),
                no_block.clone(),
            ),
            Kind::Mlp => (
                Mixer::None,
                Ffn::Dense {
                    ff: spec_u32("feed_forward_length", hp.ff[l])?,
                    act: Act::ReluSqr,
                },
            ),
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Plain,
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::NemotronH,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::{Act, Ffn, Mixer, Need, RopeMode};

    use super::super::names;
    use crate::arch::coverage;
    use crate::arch::synthetic::V;
    use crate::arch::{Arch, spec};

    /// Six layers: Mamba, MLP, attention, and the same again.
    const N: usize = 6;
    const EMBD: u64 = 32;
    const HEADS: u64 = 4;
    const KV: i32 = 2;
    const HEAD: u64 = 8;
    const FF: i32 = 64;
    const INNER: u64 = 32;
    const MAMBA_HEADS: u64 = 4;
    const CONV: u64 = 4;

    /// `d_state` and `n_groups` are the two widths a test varies. The rest
    /// of the geometry is the constants above.
    struct Geo {
        d_state: u64,
        n_groups: u64,
    }

    fn geo() -> Geo {
        Geo {
            d_state: 128,
            n_groups: 4,
        }
    }

    /// Input-projection columns, conv columns, and the grouped norm's rows.
    fn products(g: &Geo) -> (u64, u64, u64) {
        let groups_state = 2 * g.n_groups * g.d_state;
        (
            2 * INNER + groups_state + MAMBA_HEADS,
            INNER + groups_state,
            INNER / g.n_groups,
        )
    }

    fn keys(g: &Geo) -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(N as u32)),
            ("embedding_length", V::U32(EMBD as u32)),
            ("context_length", V::U32(128)),
            ("vocab_size", V::U32(EMBD as u32)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.key_length", V::U32(HEAD as u32)),
            ("attention.value_length", V::U32(HEAD as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("rope.dimension_count", V::U32(4)),
            (
                "attention.head_count_kv",
                V::I32s(vec![0, 0, KV, 0, 0, KV]),
            ),
            ("feed_forward_length", V::I32s(vec![0, FF, 0, 0, FF, 0])),
            ("ssm.conv_kernel", V::U32(CONV as u32)),
            ("ssm.inner_size", V::U32(INNER as u32)),
            ("ssm.state_size", V::U32(g.d_state as u32)),
            ("ssm.group_count", V::U32(g.n_groups as u32)),
            ("ssm.time_step_rank", V::U32(MAMBA_HEADS as u32)),
        ]
    }

    fn one(name: String, dims: Vec<u64>, out: &mut Vec<(String, Vec<u64>)>) {
        out.push((name, dims));
    }

    /// Layer `l`'s tensors for `kind` (`m`, `a` or `p`), conv bias included
    /// when `bias`.
    fn layer(l: usize, kind: char, g: &Geo, bias: bool, out: &mut Vec<(String, Vec<u64>)>) {
        let (proj, conv, norm) = products(g);
        let e = EMBD;
        one(names::attn_norm(l), vec![e], out);
        match kind {
            'm' => {
                let heads = MAMBA_HEADS;
                one(format!("blk.{l}.{}", names::SSM_A), vec![1, heads], out);
                one(format!("blk.{l}.{}", names::SSM_D), vec![1, heads], out);
                one(
                    format!("blk.{l}.{}", names::SSM_DT_BIAS),
                    vec![heads],
                    out,
                );
                one(
                    format!("blk.{l}.{}", names::SSM_CONV),
                    vec![CONV, conv],
                    out,
                );
                if bias {
                    one(
                        format!("blk.{l}.{}", names::SSM_CONV_BIAS),
                        vec![conv],
                        out,
                    );
                }
                one(names::ssm_in(l), vec![e, proj], out);
                one(
                    format!("blk.{l}.{}", names::SSM_NORM),
                    vec![norm, g.n_groups],
                    out,
                );
                one(format!("blk.{l}.{}", names::SSM_OUT), vec![INNER, e], out);
            }
            'a' => {
                let (h, kv, d) = (HEADS, u64::from(KV as u32), HEAD);
                one(names::attn_q(l), vec![e, h * d], out);
                one(format!("blk.{l}.{}", names::ATTN_K), vec![e, kv * d], out);
                one(format!("blk.{l}.{}", names::ATTN_V), vec![e, kv * d], out);
                one(
                    format!("blk.{l}.{}", names::ATTN_OUTPUT),
                    vec![h * d, e],
                    out,
                );
            }
            'p' => {
                let ff = u64::from(FF as u32);
                one(names::ffn_up(l), vec![e, ff], out);
                one(format!("blk.{l}.{}", names::FFN_DOWN), vec![ff, e], out);
            }
            _ => panic!("kind {kind}"),
        }
    }

    fn tensors(g: &Geo, bias: bool) -> Vec<(String, Vec<u64>)> {
        let mut out = Vec::new();
        one(names::token_embd(), vec![EMBD, EMBD], &mut out);
        one(names::output(), vec![EMBD, EMBD], &mut out);
        one(names::output_norm(), vec![EMBD], &mut out);
        for (l, kind) in ['m', 'p', 'a', 'm', 'p', 'a'].into_iter().enumerate() {
            layer(l, kind, g, bias, &mut out);
        }
        out
    }

    fn load(
        tag: &str,
        arch: &str,
        kv: Vec<(&str, V)>,
        tensors: &[(String, Vec<u64>)],
    ) -> Result<crate::arch::Read, String> {
        let global = [("tokenizer.ggml.pre", V::Str("qwen2"))];
        let path = crate::arch::synthetic::header_shaped(tag, arch, &kv, &global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    fn ok(tag: &str, g: &Geo, bias: bool) -> crate::arch::Read {
        load(tag, "nemotron_h", keys(g), &tensors(g, bias)).expect("the header reads")
    }

    /// Layer 0 is Mamba-2, layer 1 the squared-ReLU MLP, layer 2 attention
    /// with no rope. The same three repeat. A Mamba layer read as attention
    /// fails this.
    #[test]
    fn a_mamba_layer_is_a_mamba2_mixer() {
        let g = geo();
        let (proj, _, _) = products(&g);
        let read = ok("nemotron-h-ok", &g, true);
        let s = &read.spec;
        assert_eq!(s.arch, models::Arch::NemotronH);
        assert_eq!(s.hidden, EMBD as u32);
        assert_eq!(s.vocab, EMBD as u32);
        assert_eq!(s.ctx_train, 128);
        assert_eq!(s.rms_eps.to_bits(), 0x3727_c5ac);
        assert!(s.mtp.is_empty());
        assert!(read.defaults.is_empty());
        assert_eq!(s.layers.len(), N);
        let kind = |l: usize| match &s.layers[l].mixer {
            Mixer::Mamba2(m) => {
                assert_eq!(m.d_state, 128);
                assert_eq!(m.d_inner, INNER as u32);
                assert_eq!(m.n_heads, MAMBA_HEADS as u32);
                assert_eq!(m.head_dim, (INNER / MAMBA_HEADS) as u32);
                assert_eq!(m.n_groups, g.n_groups as u32);
                assert_eq!(m.d_conv, CONV as u32);
                assert_eq!(m.d_in_proj, proj as u32);
                assert!(matches!(
                    s.layers[l].ffn,
                    Ffn::Dense {
                        ff: 0,
                        act: Act::ReluSqr
                    }
                ));
                'm'
            }
            Mixer::None => {
                assert!(matches!(
                    s.layers[l].ffn,
                    Ffn::Dense {
                        ff: 64,
                        act: Act::ReluSqr
                    }
                ));
                'p'
            }
            Mixer::Gqa(q) => {
                assert_eq!(q.rope.mode, RopeMode::None);
                assert_eq!(q.rope.dims, 0);
                assert_eq!((q.heads, q.kv_heads, q.head_dim), (4, 2, 8));
                'a'
            }
            other => panic!("layer {l}: {other:?}"),
        };
        assert_eq!((0..N).map(kind).collect::<String>(), "mpampa");
        let stores = runtime::state::stores(&s.layers[0]);
        assert!(
            stores.iter().any(|(store, rule)| {
                matches!(
                    (store, rule),
                    (
                        runtime::state::Store::Conv,
                        runtime::state::StoreRule::Window { slots: 3 }
                    )
                )
            }),
            "{stores:?}"
        );
        assert!(
            stores
                .iter()
                .any(|(store, rule)| matches!(
                    (store, rule),
                    (runtime::state::Store::State, runtime::state::StoreRule::Recurrent)
                )),
            "{stores:?}"
        );
        assert!(runtime::state::stores(&s.layers[1]).is_empty());
    }

    /// An attention layer emits no rope need. Emitting one fails this.
    #[test]
    fn an_attention_layer_needs_no_rope() {
        let read = ok("nemotron-h-rope", &geo(), true);
        let needs = models::needs(&read.spec);
        assert!(
            needs.iter().all(|(n, _)| !matches!(n, Need::QkRope { .. })),
            "{needs:?}"
        );
        let items = coverage::check(&read.spec, &read.tensors);
        assert!(
            items
                .iter()
                .any(|u| u.feature == "a layer program for nemotron_h"),
            "{items:?}"
        );
        assert!(
            items.iter().any(|u| u.feature.starts_with("Mamba-2 mixer")),
            "{items:?}"
        );
        assert!(
            items
                .iter()
                .any(|u| u.feature.starts_with("squared-ReLU dense layer")),
            "{items:?}"
        );
        assert!(
            items.iter().all(|u| !u.feature.contains("rope")),
            "{items:?}"
        );
        assert!(
            items.iter().all(|u| !u.feature.starts_with("dense SwiGLU")),
            "{items:?}"
        );
    }

    /// The grouped norm is `d_inner / n_groups` rows, not the head dim. Here
    /// the two differ (16 and 8); a norm shaped as the head dim is refused.
    #[test]
    fn the_grouped_norm_is_inner_over_groups() {
        let g = Geo {
            d_state: 128,
            n_groups: 2,
        };
        let read = ok("nemotron-h-groups", &g, true);
        match &read.spec.layers[0].mixer {
            Mixer::Mamba2(m) => {
                assert_eq!(m.n_groups, 2);
                assert_eq!(m.head_dim, 8);
            }
            other => panic!("{other:?}"),
        }
        let mut bad = tensors(&g, true);
        let norm = format!("blk.0.{}", names::SSM_NORM);
        let row = bad.iter_mut().find(|(n, _)| *n == norm).expect("the norm");
        row.1 = vec![INNER / MAMBA_HEADS, g.n_groups];
        let err = load("nemotron-h-norm", "nemotron_h", keys(&g), &bad).expect_err("a head-dim norm");
        assert!(err.contains(&norm), "{err}");
    }

    /// `nemotron_h_moe`, and any other `nemotron*` string, is not this reader.
    #[test]
    fn nemotron_h_moe_is_unknown() {
        for arch in ["nemotron_h_moe", "nemotron", "nemotron_x"] {
            assert!(
                Arch::from_name(arch).is_err(),
                "{arch} is a known architecture"
            );
            let err = load(arch, arch, vec![], &[]).expect_err(arch);
            assert_eq!(err, format!("unsupported architecture \"{arch}\""));
        }
        assert_eq!(Arch::from_name("nemotron_h").unwrap(), Arch::NemotronH);
    }

    /// Each `ssm.*` key is required by name. Dropping the check for one of
    /// them lets this header read.
    #[test]
    fn a_missing_ssm_key_is_named() {
        let g = geo();
        let tensors = tensors(&g, true);
        for key in [
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.group_count",
            "ssm.time_step_rank",
        ] {
            let kv: Vec<_> = keys(&g).into_iter().filter(|(k, _)| *k != key).collect();
            let err = load("nemotron-h-ssm", "nemotron_h", kv, &tensors).expect_err(key);
            assert!(
                err.contains(key) && err.contains("absent"),
                "{key}: {err}"
            );
        }
    }

    /// A state width other than 128 is refused by name, before the tensors
    /// are walked. Accepting 96 lets this header read: its tensors are built
    /// at 96.
    #[test]
    fn a_state_other_than_128_is_named() {
        let g = Geo {
            d_state: 96,
            n_groups: 4,
        };
        let err = load("nemotron-h-state", "nemotron_h", keys(&g), &tensors(&g, true))
            .expect_err("state 96");
        assert!(
            err.contains("ssm.state_size") && err.contains("is 96, not 128"),
            "{err}"
        );
    }

    /// A layer whose marker tensor disagrees with the arrays is refused by
    /// name, naming the layer and both kinds.
    #[test]
    fn a_layer_whose_tensors_disagree_is_named() {
        let g = geo();
        let mut tensors = tensors(&g, true);
        tensors.retain(|(n, _)| !n.starts_with("blk.0."));
        layer(0, 'a', &g, true, &mut tensors);
        let err = load(
            "nemotron-h-kind",
            "nemotron_h",
            keys(&g),
            &tensors,
        )
        .expect_err("layer 0 as attention");
        assert!(
            err.contains("attention.head_count_kv")
                && err.contains("layer 0")
                && err.contains("Mamba")
                && err.contains("Attention")
                && err.contains("feed_forward_length"),
            "{err}"
        );
    }

    /// The conv bias is not required.
    #[test]
    fn the_conv_bias_may_be_absent() {
        let read = ok("nemotron-h-bias", &geo(), false);
        assert!(matches!(read.spec.layers[0].mixer, Mixer::Mamba2(_)));
        assert!(
            !read
                .tensors
                .tensors
                .iter()
                .any(|t| t.name.contains("ssm_conv1d.bias"))
        );
    }

    /// `rope.dimension_count` is required, and it is not put on the attention.
    #[test]
    fn rope_dimension_count_is_required_and_not_applied() {
        let g = geo();
        let kv: Vec<_> = keys(&g)
            .into_iter()
            .filter(|(k, _)| *k != "rope.dimension_count")
            .collect();
        let err = load("nemotron-h-nord", "nemotron_h", kv, &tensors(&g, true))
            .expect_err("no rope dims");
        assert!(err.contains("rope.dimension_count"), "{err}");
    }

    /// A per-layer array shorter than `block_count` is refused by name.
    #[test]
    fn a_short_per_layer_array_is_named() {
        let g = geo();
        let mut kv = keys(&g);
        let row = kv
            .iter_mut()
            .find(|(k, _)| *k == "attention.head_count_kv")
            .expect("the kv array");
        *row = (
            "attention.head_count_kv",
            V::I32s(vec![0, 0, KV, 0, 0]),
        );
        let err = load("nemotron-h-short", "nemotron_h", kv, &tensors(&g, true))
            .expect_err("short array");
        assert!(
            err.contains("attention.head_count_kv") && err.contains("5 values"),
            "{err}"
        );
    }

    /// A tensor no table holds is refused by name.
    #[test]
    fn an_unknown_tensor_is_named() {
        let g = geo();
        let mut tensors = tensors(&g, true);
        tensors.push(("blk.0.ffn_gate_exps.weight".to_string(), vec![1, 1]));
        let err = load("nemotron-h-extra", "nemotron_h", keys(&g), &tensors).expect_err("extra");
        assert!(err.contains("ffn_gate_exps.weight"), "{err}");
    }

    /// A next-token layer count is refused by name. Absent, the draft list
    /// is empty (the success test).
    #[test]
    fn a_nonzero_nextn_count_is_named() {
        let g = geo();
        let mut kv = keys(&g);
        kv.push(("nextn_predict_layers", V::U32(1)));
        let err = load("nemotron-h-nextn", "nemotron_h", kv, &tensors(&g, true)).expect_err("nextn");
        assert!(
            err.contains("nextn_predict_layers") && err.contains("is 1"),
            "{err}"
        );
    }
}
