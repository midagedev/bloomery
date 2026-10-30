//! The raon1 reader's [`ModelSpec`]. An RDA layer is a [`Mixer::DeltaRule`] of
//! `attention.head_count` key and value heads of `kda.head_dim` values; a latent layer caches
//! the `kv_lora_rank` latent and the `rope.dimension_count` key values no rope turns a position,
//! absorbs the per-head up projections (`attn_k_b`, `attn_v_b`) and gates its heads' outputs by
//! a projection of the layer input (`src/models/raon1.cpp`). Every trunk layer owns the
//! hyper-connection mix of its two sublayers, and the streams collapse to their mean before the
//! head. The dense layers and the experts take `situ`; the experts' router is the common noaux
//! owner's. The tool calls are [`TOOLS`], the `<think>` span is read.

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Collapse, DeltaKind, DeltaRule, Ffn, HcKind, HcSpec, Latent, LatentOut,
    LatentUp, LayerSpec, Mixer, ModelSpec, OutGate, ReasoningFormat, Residual, Source, ToolFormat,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// The tool-call markup the family's template teaches: `<tool_call>` with `<function=…>` and
/// `<parameter=…>`.
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::QwenXml);

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(ReasoningFormat::THINK))?;
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` was read from, with the chat surface `chat`.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let act = Act::Situ {
        gate_beta: hp.situ.gate_beta,
        value_beta: hp.situ.value_beta,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = match hp.kind(l) {
            Kind::Rda => Mixer::DeltaRule(DeltaRule {
                kind: DeltaKind::Rda { gate: hp.gate },
                k_heads: spec_u32("attention.head_count", hp.n_head)?,
                v_heads: spec_u32("attention.head_count", hp.n_head)?,
                d: spec_u32("kda.head_dim", hp.kda.head_dim)?,
                conv: spec_u32("ssm.conv_kernel", hp.kda.conv)?,
            }),
            Kind::Mla => {
                let m = &hp.mla;
                Mixer::Latent(Latent {
                    heads: spec_u32("attention.head_count", hp.n_head)?,
                    q_lora: spec_u32("attention.q_lora_rank", m.q_lora)?,
                    latent: spec_u32("attention.kv_lora_rank", m.kv_lora)?,
                    up: LatentUp::Absorbed {
                        qk: spec_u32("attention.key_length_mla", m.head_k)?,
                        v: spec_u32("attention.value_length_mla", m.head_v)?,
                    },
                    rope: None,
                    q_head_norm: false,
                    ssmax: false,
                    pe: spec_u32("rope.dimension_count", m.pe)?,
                    cache: Source::Own(()),
                    out_gate: OutGate::Proj,
                    out: LatentOut::Plain,
                    window: None,
                    sinks: false,
                    compress: None,
                    select: None,
                })
            }
        };
        let ffn = if hp.dense(l) {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.dense_ff)?,
                act,
            }
        } else {
            hp.experts.ffn(act)?
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Hc,
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::Raon1,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: Some(HcSpec {
            streams: spec_u32("hyper_connection.count", hp.hc.streams)?,
            kind: HcKind::Chart {
                eps: hp.hc.eps,
                magnitude: hp.hc.magnitude,
                collapse: Collapse::Mean,
            },
        }),
        engram: None,
        encoder: None,
        logit_epilogue: None,
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::{
        Act, Collapse, DeltaKind, Ffn, HcKind, LatentUp, Mixer, OutGate, Residual, Score, Selector,
        ToolFormat,
    };

    use super::super::hparams::tests::{keys, tensors};
    use crate::arch::coverage;
    use crate::arch::synthetic::{V, header_shaped};

    /// The small header of the `hparams` tests with `tokenizer.ggml.pre` `raon` and a chat
    /// template: its read and the coverage check's items.
    fn read(tag: &str) -> (super::Read, Vec<String>) {
        let global = [
            ("tokenizer.ggml.pre", V::Str("raon")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "raon1", &keys(), &global, &tensors());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        let read = read.expect("the header reads");
        let items = coverage::check(&read.spec, &read.tensors)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        (read, items)
    }

    /// The description is the table's: a delta rule of 4 heads of 16 with the erase and write
    /// gates apart and the sigmoid decay bounded at -5 on layers 0, 1, 3 and 4; a latent layer on
    /// 2 and 5 caching 32 latent values and 8 unrotated ones, absorbing 24-wide queries into 16-wide
    /// values, gated by a projection; the first layer a dense `situ` block and the rest the
    /// sigmoid router over `situ` experts beside an ungated shared one; every layer on the
    /// hyper-connections, three streams, the chart mix. Mutant: the write gate dropped from the
    /// delta kind, the gate on the wrong layer kind, or the streams collapsed another way.
    #[test]
    fn the_description_is_the_tables() {
        let (read, _) = read("raon1-spec");
        let spec = &read.spec;
        assert_eq!(spec.arch, models::Arch::Raon1);
        assert_eq!((spec.hidden, spec.vocab, spec.ctx_train), (64, 2, 4096));
        assert!(spec.mtp.is_empty() && spec.engram.is_none() && spec.encoder.is_none());
        let hc = spec.hc.expect("hyper-connections");
        assert_eq!(hc.streams, 3);
        assert_eq!(
            hc.kind,
            HcKind::Chart {
                eps: 1e-5,
                magnitude: 2.0,
                collapse: Collapse::Mean
            }
        );
        let act = Act::Situ {
            gate_beta: 4.0,
            value_beta: 25.0,
        };
        assert_eq!(spec.layers.len(), 6);
        for (l, layer) in spec.layers.iter().enumerate() {
            assert_eq!(layer.residual, Residual::Hc, "layer {l}");
            assert!(layer.extras.is_empty());
            match &layer.mixer {
                Mixer::DeltaRule(d) => {
                    assert!([0, 1, 3, 4].contains(&l), "layer {l}");
                    assert_eq!(
                        d.kind,
                        DeltaKind::Rda {
                            gate: models::KdaGate::Sigmoid { lower_bound: -5.0 }
                        }
                    );
                    assert_eq!((d.k_heads, d.v_heads, d.d, d.conv), (4, 4, 16, 4));
                }
                Mixer::Latent(a) => {
                    assert!([2, 5].contains(&l), "layer {l}");
                    assert_eq!((a.heads, a.q_lora, a.latent, a.pe), (4, 24, 32, 8));
                    assert_eq!(a.up, LatentUp::Absorbed { qk: 24, v: 16 });
                    assert_eq!(a.out_gate, OutGate::Proj);
                    assert!(a.rope.is_none() && !a.q_head_norm && !a.sinks);
                    assert!(a.window.is_none() && a.compress.is_none());
                    assert_eq!(a.select, None::<Selector>);
                }
                other => panic!("layer {l}: {}", other.what()),
            }
            if l == 0 {
                assert_eq!(layer.ffn, Ffn::Dense { ff: 48, act });
                continue;
            }
            let Ffn::Moe(m) = &layer.ffn else {
                panic!("layer {l} is not routed");
            };
            assert_eq!(
                (m.experts, m.top_k, m.expert_ff, m.latent),
                (8, 2, 24, None)
            );
            assert_eq!(m.act, act);
            assert_eq!(m.router.score, Score::Sigmoid);
            assert!(m.router.bias && !m.router.hash);
            assert_eq!(m.router.norm, models::RouterNorm::SelectedSum);
            assert_eq!(m.router.scale, 1.0);
            let shared = m.shared.expect("a shared expert");
            assert_eq!(
                (shared.ff, shared.act, shared.sigmoid_gate),
                (24, act, false)
            );
        }
        assert_eq!(spec.chat.tools, Some(ToolFormat::QwenXml));
        assert_eq!(spec.chat.reasoning, Some(models::ReasoningFormat::THINK));
    }

    /// The body runs every kind the file names, so only the small header's widths are listed:
    /// each kind's coverage row holds Raon's own widths, and a file of other widths is refused by
    /// name rather than run wrong. The kinds themselves (the unrotated latent values, the latent
    /// output gate, the layer program), the pre-tokenizer `raon` and the tool-call parser are not
    /// among them. Mutant: `Program::Raon1Body` left out of a kind's row lists that kind.
    #[test]
    fn only_the_small_widths_are_listed() {
        let (_, items) = read("raon1-coverage");
        let listed = |needle: &str| items.iter().any(|f| f.contains(needle));
        for needle in [
            "delta rule RDA: d 16",
            "dense situ layer: ff 48",
            "chart-constrained hyper-connections of 3 streams",
            "latent attention 32 x rope 0",
            "router: sigmoid, 8 experts",
            "shared expert, ff 24",
        ] {
            assert!(listed(needle), "{needle:?} not listed: {items:#?}");
        }
        for needle in [
            "unrotated key values",
            "latent attention output gate",
            "a layer program for raon1",
        ] {
            assert!(!listed(needle), "{needle:?} listed: {items:#?}");
        }
        assert!(!listed("pre-tokenizer"), "{items:#?}");
        assert!(!listed("tool-call parser"), "{items:#?}");
    }
}
