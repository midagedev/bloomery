//! The raon1 reader's [`ModelSpec`], both dialects'. An RDA layer is a [`Mixer::DeltaRule`] of
//! its own `attention.head_count` key and value heads of `kda.head_dim` values; a latent layer
//! caches the `kv_lora_rank` latent and the `rope.dimension_count` key values no rope turns a
//! position, absorbs the per-head up projections (`attn_k_b`, `attn_v_b`) and gates its heads'
//! outputs by a projection of the layer input (`src/models/raon1.cpp`,
//! `src/models/raon1nano.cpp`). Every trunk layer owns the hyper-connection mix of its two
//! sublayers — the Mini file's scale scalars and the Nano file's folded chart one mix alike —
//! and the streams collapse to their mean before the head. The trunk's feed-forward side is the
//! file's: the Mini file's `situ` dense lead and sigmoid noaux router over `situ` experts, the
//! Nano file's one SwiGLU block in every layer with the per-layer input gated into the streams
//! after it ([`Extra::PerLayerEmbed`]). The next-token blocks the Nano file carries in their own
//! file are described ([`ModelSpec::mtp`]) and run by no program. The tool calls are [`TOOLS`],
//! the `<think>` span is read.

use gguf::Split;
use models::{
    ChatSpec, Collapse, DeltaKind, DeltaRule, Extra, Ffn, HcKind, HcSpec, LayerSpec, Mixer,
    ModelSpec, ReasoningFormat, Residual, ToolFormat,
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
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = match hp.kind(l) {
            Kind::Rda => Mixer::DeltaRule(DeltaRule {
                kind: DeltaKind::Rda { gate: hp.gate },
                k_heads: spec_u32("attention.head_count", hp.head(l))?,
                v_heads: spec_u32("attention.head_count", hp.head(l))?,
                d: spec_u32("kda.head_dim", hp.kda.head_dim)?,
                conv: spec_u32("ssm.conv_kernel", hp.kda.conv)?,
            }),
            Kind::Mla => Mixer::Latent(hp.mla.latent(hp.head(l))?),
        };
        let ffn = if hp.dense(l) {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.dense_ff)?,
                act: hp.act,
            }
        } else {
            hp.experts
                .as_ref()
                .expect("a routed layer: the read refused a file whose router keys are absent")
                .ffn(hp.act)?
        };
        let extras = match (hp.ple, l < hp.n_trunk) {
            (Some(width), true) => vec![Extra::PerLayerEmbed {
                width: spec_u32("embedding_length_per_layer_input", width)?,
            }],
            _ => Vec::new(),
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Hc,
            extras,
        })
    };
    let layers = (0..hp.n_trunk).map(&layer).collect::<Result<Vec<_>, _>>()?;
    let mtp = (hp.n_trunk..hp.n_layer)
        .map(&layer)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ModelSpec {
        arch: hp.arch,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp,
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
        Act, Collapse, DeltaKind, Extra, Ffn, HcKind, LatentUp, Mixer, OutGate, Residual, Score,
        Selector, ToolFormat,
    };

    use super::super::hparams::tests::{keys, keys_nano, mtp_nano, tensors, tensors_nano};
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

    /// The small Nano header of the `hparams` tests with a chat surface: its read.
    fn read_nano_file(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> super::Read {
        let global = [
            ("tokenizer.ggml.pre", V::Str("raon")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "raon1nano", kv, &global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read.unwrap_or_else(|e| panic!("{tag} does not read: {e}"))
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
            assert!(layer.extras.is_empty(), "layer {l}");
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

    /// The Nano description is its own table: every layer a dense SwiGLU block of the row's
    /// activation with a per-layer input site of the key's width on it, the delta layers 4 heads
    /// and the latent ones 8 of their own, the head tied, no next-token layer in the trunk's
    /// file. Mutant: the activation read from the situ keys, the extras left empty, or the
    /// latent layer's head count read as the delta one's.
    #[test]
    fn the_nano_description_is_the_tables() {
        let read = read_nano_file("raon1nano-spec", &keys_nano(), &tensors_nano());
        let spec = &read.spec;
        assert_eq!(spec.arch, models::Arch::Raon1Nano);
        assert_eq!((spec.hidden, spec.vocab, spec.ctx_train), (64, 2, 4096));
        assert_eq!(spec.rms_eps, 1e-6);
        assert_eq!(spec.layers.len(), 6);
        assert!(spec.mtp.is_empty());
        let hc = spec.hc.expect("hyper-connections");
        assert_eq!(
            hc.kind,
            HcKind::Chart {
                eps: 1e-5,
                magnitude: 1.0,
                collapse: Collapse::Mean
            }
        );
        for (l, layer) in spec.layers.iter().enumerate() {
            assert_eq!(
                layer.extras,
                vec![Extra::PerLayerEmbed { width: 16 }],
                "layer {l}"
            );
            assert_eq!(
                layer.ffn,
                Ffn::Dense {
                    ff: 48,
                    act: Act::SwiGlu { limit: None }
                },
                "layer {l}"
            );
            match &layer.mixer {
                Mixer::DeltaRule(d) => {
                    assert!([0, 1, 3, 4].contains(&l), "layer {l}");
                    assert_eq!((d.k_heads, d.v_heads), (4, 4), "layer {l}");
                }
                Mixer::Latent(a) => {
                    assert!([2, 5].contains(&l), "layer {l}");
                    assert_eq!(a.heads, 8, "layer {l}");
                }
                other => panic!("layer {l}: {}", other.what()),
            }
        }
    }

    /// The MTP file's description holds the seventh block in `mtp`, a latent layer of the dense
    /// block and the chart mix and no per-layer input, and the trunk's six unchanged. Mutant: the
    /// mtp list left empty, or the next-token layer given a per-layer input.
    #[test]
    fn the_nano_mtp_block_is_described() {
        let (kv, t) = mtp_nano();
        let read = read_nano_file("raon1nano-mtp", &kv, &t);
        let spec = &read.spec;
        assert_eq!(spec.layers.len(), 6);
        let [mtp] = spec.mtp.as_slice() else {
            panic!("one next-token layer, got {}", spec.mtp.len());
        };
        assert!(
            mtp.extras.is_empty(),
            "the next-token layer has no per-layer input"
        );
        let Mixer::Latent(a) = &mtp.mixer else {
            panic!("the next-token layer is a latent one");
        };
        assert_eq!((a.heads, a.latent, a.pe), (8, 32, 8));
        assert_eq!(
            mtp.ffn,
            Ffn::Dense {
                ff: 48,
                act: Act::SwiGlu { limit: None }
            }
        );
        assert_eq!(mtp.residual, Residual::Hc);
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

    /// The raon1 program runs the Nano file: `program_of` names it, and the
    /// coverage check lists none of its parts — the layer program, the dense
    /// SwiGLU block and the folded chart covered — the small header's own
    /// items the widths no launch runs (the delta rule's and the latent
    /// layer's, as the Mini file's small header lists them, and a per-layer
    /// input of 16 values, the norm's 32). Mutants: the `program_of` arm
    /// given `None`, which lists the layer program, and the dense SwiGLU
    /// row dropped, which lists the block.
    #[test]
    fn the_program_runs_the_nano_file() {
        use crate::placement::accepts::{Program, program_of};

        let read = read_nano_file("raon1nano-coverage", &keys_nano(), &tensors_nano());
        assert_eq!(
            program_of(models::Arch::Raon1Nano),
            Some(Program::Raon1Body)
        );
        assert_eq!(program_of(models::Arch::Raon1), Some(Program::Raon1Body));
        let items: Vec<String> = coverage::check(&read.spec, &read.tensors)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        let listed = |needle: &str| items.iter().any(|f| f.contains(needle));
        assert!(!listed("a layer program"), "{items:#?}");
        assert!(!listed("dense SwiGLU layer"), "{items:#?}");
        for needle in [
            "delta rule RDA: d 16",
            "latent attention 32 x rope 0",
            "a per-layer input of 16 values",
        ] {
            assert!(listed(needle), "{needle:?} not listed: {items:#?}");
        }
    }
}
